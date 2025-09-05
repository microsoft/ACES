"""
Docker-based SQL executor for executing SQL queries in isolated containers.

This module provides a secure SQL executor that accepts SQL query strings over the MCP protocol
and executes them in Docker containers. It's designed to work with the Excytin threat investigation
environment and other SQL-based execution environments.
"""

import logging
from typing import Any, Dict, List, Optional

from ....base import CommandResult
from ...base import Parameter, ParameterType, ValidationResult
from ...exceptions import SandboxExecutionError
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ...utils.security_validator import SecurityValidator
from ..docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class SQLExecutor(DockerExecutor):
    """
    Docker-based SQL executor for secure database query execution.

    This executor provides a unified interface for executing SQL queries
    in isolated Docker containers with comprehensive security validation.
    Queries are executed via the mysql client or other DB clients.

    Execution Features:
    - Docker container isolation
    - Timeout enforcement
    - Connection string management
    - Episode-based container management
    - Query validation
    """

    _executor_metadata = {
        "name": "sql",
        "description": "Execute SQL queries in the SABER sandbox environment.",
    }

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration for SQL executor.

        Returns:
            Dictionary containing SQL executor default configuration
        """
        return {
            "timeout": 60.0,
            "max_rows": 1000,
            "allow_schema_queries": True,
        }

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "SQLExecutor":
        """
        Create SQL executor with standardized configuration interface.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: SQL-specific configuration dictionary
            additional_params: Additional parameters (e.g., sql_config)
            **kwargs: Additional keyword arguments

        Returns:
            Configured SQL executor instance
        """
        merged_kwargs = {**kwargs}

        # Extract SQL-specific parameters from additional_params
        sql_config = None
        if additional_params:
            sql_config = additional_params.get("sql_config")
            # Remove from kwargs since it's a specific parameter
            if "sql_config" in additional_params:
                additional_params = {k: v for k, v in additional_params.items() if k != "sql_config"}
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, sql_config=sql_config, **merged_kwargs)

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        config: Optional[Dict[str, Any]] = None,
        sql_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Docker SQL executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: SQL configuration dictionary
            sql_config: SQL-specific configuration (connection string, etc.)
            **kwargs: Additional arguments passed to parent

        Raises:
            SandboxExecutionError: If sandbox_manager is None or invalid
        """
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

        # Initialize SQL configuration
        self._sql_config = sql_config or {}

        # Extract connection string or defaults
        self._connection_string = self._sql_config.get("connection_string", "mysql://root:admin@localhost:3306/mysql")
        self._allow_schema_queries = self._config.get("allow_schema_queries", True)
        self._max_rows = self._config.get("max_rows", 1000)

        # Initialize security validator for SQL queries
        self._security_validator = SecurityValidator()

    def setup_parameters(self, config: Dict[str, Any]) -> None:
        """Set up SQL executor parameters."""
        # Add parameter for the SQL query
        self.add_parameter(
            Parameter(
                name="query",
                type=ParameterType.STRING,
                description="SQL query to execute in the database",
                required=True,
            )
        )

        # Optional parameters
        self.add_parameter(
            Parameter(
                name="connection_string",
                type=ParameterType.STRING,
                description="Database connection string (overrides executor default)",
                required=False,
            )
        )

        # Get max_rows from config instead of using self._max_rows since the attribute may not be initialized yet
        max_rows = config.get("max_rows", 1000)
        self.add_parameter(
            Parameter(
                name="max_rows",
                type=ParameterType.INTEGER,
                description="Maximum number of rows to return",
                required=False,
                default=max_rows,
            )
        )

    def parse_connection_string(self, connection_string: str) -> Dict[str, str]:
        """
        Parse a connection string into components.

        Args:
            connection_string: Database connection string in format "mysql://user:pass@host:port/dbname"

        Returns:
            Dictionary with connection components

        Raises:
            ValueError: If connection string format is invalid
        """
        try:
            # Parse the connection string
            if "://" not in connection_string:
                raise ValueError("Connection string must include protocol (e.g., mysql://)")

            protocol, rest = connection_string.split("://", 1)

            # Parse authentication and host information
            if "@" in rest:
                auth, host_part = rest.split("@", 1)
            else:
                auth = "root:admin"  # Default credentials
                host_part = rest

            # Parse username and password
            if ":" in auth:
                username, password = auth.split(":", 1)
            else:
                username = auth
                password = ""  # Empty password

            # Parse host, port, and database
            if "/" in host_part:
                host_port, database = host_part.split("/", 1)
            else:
                host_port = host_part
                database = ""  # No database specified

            # Parse host and port
            if ":" in host_port:
                host, port = host_port.split(":", 1)
            else:
                host = host_port
                port = "3306"  # Default MySQL port

            return {
                "protocol": protocol,
                "username": username,
                "password": password,
                "host": host,
                "port": port,
                "database": database,
            }
        except Exception as e:
            raise ValueError(f"Invalid connection string format: {str(e)}")

    def build_mysql_command(self, query: str, connection_info: Dict[str, str]) -> List[str]:
        """
        Build the mysql client command for executing the query.

        Args:
            query: SQL query to execute
            connection_info: Connection information dictionary

        Returns:
            List of command arguments ready for Docker execution
        """
        # Format the query: add semicolon if missing and wrap in quotes
        if not query.strip().endswith(";"):
            query = f"{query.strip()};"

        # Build mysql command with proper escaping
        mysql_cmd = [
            "mysql",
            "-h",
            connection_info["host"],
            "-P",
            connection_info["port"],
            "-u",
            connection_info["username"],
        ]

        # Add password if provided
        if connection_info["password"]:
            mysql_cmd.extend([f"-p{connection_info['password']}"])

        # Add database if provided
        if connection_info["database"]:
            mysql_cmd.extend([connection_info["database"]])

        # Add query execution flags
        mysql_cmd.extend(["--table", "-e", query])  # Table format output  # Execute query

        return mysql_cmd

    def validate_sql_query(self, query: str) -> ValidationResult:
        """
        Validate SQL query for syntax and security.

        Args:
            query: SQL query string to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult.success()

        if not query or not query.strip():
            result.add_error("SQL query cannot be empty")
            return result

        # Basic security checks
        dangerous_patterns = [
            "DROP DATABASE",
            "DROP SCHEMA",
            "DROP USER",
            "CREATE USER",
            "GRANT ",
            "REVOKE ",
            "ALTER USER",
            "SHUTDOWN",
            "TRUNCATE TABLE",
        ]

        # Check for dangerous SQL patterns
        for pattern in dangerous_patterns:
            if pattern.upper() in query.upper():
                result.add_warning(f"Potentially dangerous SQL construct detected: {pattern}")

        # Only allow schema queries if explicitly enabled
        schema_patterns = ["SHOW TABLES", "DESCRIBE ", "DESC ", "SHOW COLUMNS", "INFORMATION_SCHEMA"]

        if not self._allow_schema_queries:
            for pattern in schema_patterns:
                if pattern.upper() in query.upper():
                    result.add_error(f"Schema queries are not allowed: {pattern}")

        return result

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute SQL query in Docker container.

        This method executes the SQL query using the appropriate database client
        in the container (e.g., mysql client for MySQL databases).

        Args:
            parameters: Tool parameters (must include 'query')
            context: Execution context including episode_id

        Returns:
            CommandResult with execution results

        Note:
            Security validation is performed at this executor level before execution.
            All execution happens in Docker containers.
        """
        try:
            # Validate parameters including SQL validation
            validation_result = self.validate_parameters(parameters)
            if not validation_result.valid:
                return CommandResult.error_result(
                    error=f"Parameter validation failed: {', '.join(validation_result.errors)}"
                )

            # Log any security warnings
            if validation_result.warnings:
                for warning in validation_result.warnings:
                    logger.warning(f"SQL security warning: {warning}")

            # Extract episode ID from context
            episode_id = context.get("episode_id")
            if not episode_id:
                raise SandboxExecutionError("episode_id required in context for SQL execution")

            # Get Docker environment for episode
            environment = self.get_episode_environment(episode_id)
            timeout = int(self.get_timeout())

            # Get SQL query
            query = parameters.get("query", "").strip()

            # Get connection string (parameter overrides default)
            connection_string = parameters.get("connection_string", self._connection_string)

            # Parse connection string
            connection_info = self.parse_connection_string(connection_string)

            # Build command based on database type
            if connection_info["protocol"].lower() in ["mysql", "mariadb"]:
                command_args = self.build_mysql_command(query, connection_info)
            else:
                return CommandResult.error_result(f"Unsupported database type: {connection_info['protocol']}")

            # Log query execution
            logger.warning(
                f"🔍 SQL QUERY START: episode={episode_id}, timeout={timeout}s, "
                f"database={connection_info['database']}, "
                f"query='{query[:100]}{'...' if len(query) > 100 else ''}'"
            )

            # Execute the command
            try:
                result = await environment.execute_command(command=command_args, timeout=timeout)
                logger.warning(
                    f"🔍 SQL QUERY SUCCESS: episode={episode_id}, exit_code={result.exit_code}, "
                    f"execution_time={result.execution_time:.2f}s"
                )
            except Exception as e:
                if "timed out" in str(e).lower():
                    logger.warning(
                        f"🔍 SQL QUERY TIMEOUT: episode={episode_id}, timeout={timeout}s, "
                        f"query='{query[:50]}...' - {str(e)}"
                    )
                else:
                    logger.warning(f"🔍 SQL QUERY ERROR: episode={episode_id}, error='{str(e)}'")
                raise

            # Parse output
            tool_result = self.parse_output(result.stdout, result.stderr, result.exit_code, query)

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "episode_id": episode_id,
                    "execution_time": result.execution_time,
                    "query": query,
                    "database": connection_info["database"],
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"SQL query execution error: {e}")
            return CommandResult.error_result(f"SQL query execution failed: {str(e)}")

    def parse_output(self, stdout: str, stderr: str, return_code: int, query: str) -> CommandResult:
        """
        Parse SQL query output into a structured result.

        Args:
            stdout: Standard output from the command
            stderr: Standard error from the command
            return_code: Process exit code
            query: Original SQL query

        Returns:
            CommandResult with structured output data
        """
        # Determine if query was successful
        success = return_code == 0

        # Create result data structure
        result_data = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": return_code,
            "return_code": return_code,  # Keep both for backward compatibility
            "success": success,
            "output": stdout if success else stderr,  # Primary output
            "query": query,
        }

        # Add metadata about execution
        metadata = {
            "command_type": "sql_query",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
            "query_type": self._get_query_type(query),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            # For failed queries, include both stdout and stderr in error message
            error_msg = f"SQL query failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"
            elif stdout.strip():
                error_msg += f". Output: {stdout.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    def _get_query_type(self, query: str) -> str:
        """
        Determine the type of SQL query.

        Args:
            query: SQL query string

        Returns:
            Query type as string (SELECT, INSERT, UPDATE, etc.)
        """
        query_upper = query.upper().strip()

        if query_upper.startswith("SELECT"):
            return "SELECT"
        elif query_upper.startswith("INSERT"):
            return "INSERT"
        elif query_upper.startswith("UPDATE"):
            return "UPDATE"
        elif query_upper.startswith("DELETE"):
            return "DELETE"
        elif query_upper.startswith("CREATE"):
            return "CREATE"
        elif query_upper.startswith("ALTER"):
            return "ALTER"
        elif query_upper.startswith("DROP"):
            return "DROP"
        elif query_upper.startswith("SHOW"):
            return "SHOW"
        elif query_upper.startswith("DESCRIBE") or query_upper.startswith("DESC"):
            return "DESCRIBE"
        elif query_upper.startswith("USE"):
            return "USE"
        else:
            return "UNKNOWN"

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters including SQL validation.

        Args:
            parameters: Tool parameters to validate

        Returns:
            ValidationResult with validation details
        """
        # First do basic parameter validation from parent
        basic_validation = super().validate_parameters(parameters)
        if not basic_validation.valid:
            return basic_validation

        # Extract query for SQL validation
        query = parameters.get("query", "")
        if not query or not isinstance(query, str):
            return ValidationResult.failure(["Query parameter is required and must be a string"])

        # Validate the SQL query
        sql_validation = self.validate_sql_query(query.strip())
        if not sql_validation.valid:
            return ValidationResult.failure([f"SQL validation failed: {', '.join(sql_validation.errors)}"])

        # Add any SQL warnings to the validation result
        if sql_validation.warnings:
            for warning in sql_validation.warnings:
                basic_validation.add_warning(warning)

        # If connection string is provided, validate it
        connection_string = parameters.get("connection_string")
        if connection_string:
            try:
                self.parse_connection_string(connection_string)
            except ValueError as e:
                return ValidationResult.failure([f"Invalid connection string: {str(e)}"])

        return basic_validation

    def get_security_info(self) -> Dict[str, Any]:
        """
        Get information about SQL security settings.

        Returns:
            Dictionary with security configuration
        """
        # Get Docker-specific information from parent class
        docker_info = self.get_docker_info()

        # Add SQL-specific security information
        sql_info = {
            "allow_schema_queries": self._allow_schema_queries,
            "max_rows": self._max_rows,
            "connection_type": self.parse_connection_string(self._connection_string)["protocol"],
        }

        return {**docker_info, **sql_info}


# Register this executor with the registry - must be at module level
from ..executor_registry import register_executor  # noqa: E402

register_executor("sql", SQLExecutor, "standard")
