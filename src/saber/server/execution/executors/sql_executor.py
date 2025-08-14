"""
Docker-based SQL query executor for database operations in isolated containers.

This module provides a secure SQL executor that accepts database connection parameters
and SQL queries, executing them in Docker containers with proper security validation
and result formatting.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class SqlExecutor(DockerExecutor):
    """
    Docker-based SQL executor for secure database operations.

    This executor provides SQL query capabilities including:
    - Multi-database support (PostgreSQL, MySQL, SQLite)
    - Connection parameter validation
    - SQL injection prevention
    - Query result formatting (JSON, CSV, table)
    - Transaction management
    - Query timeout and resource limits
    """

    _security_command_metadata = {
        "domain": "database",
        "name": "sql_query",
        "description": "Execute SQL queries against databases in Docker containers",
        "author": "SABER Team",
        "security_level": "high",
        "requires_validation": True,
    }

    def __init__(
        self, sandbox_manager: SandboxManager, sql_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize SQL executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            sql_config: Optional SQL-specific configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, docker_config=sql_config, **kwargs)

        self._sql_config = sql_config or {}

        # Set up supported database types
        self._supported_databases = self._sql_config.get(
            "supported_databases", ["postgresql", "mysql", "sqlite", "mariadb"]
        )

        # Set up dangerous SQL patterns (security feature)
        self._dangerous_patterns = self._sql_config.get(
            "dangerous_patterns",
            [
                r"\bdrop\s+database\b",
                r"\bdrop\s+table\b",
                r"\bdelete\s+from\b.*\bwhere\s+1\s*=\s*1\b",
                r"\btruncate\s+table\b",
                r"\balter\s+table\b.*\bdrop\b",
                r"\bgrant\b.*\ball\b",
                r"\brevoke\b",
                r"\bcreate\s+user\b",
                r"\bdrop\s+user\b",
                r"\bshutdown\b",
                r"--\s*.*",  # SQL comments that might hide malicious code
                r"/\*.*\*/",  # Block comments
            ],
        )

        # Set up allowed operations
        self._allowed_operations = self._sql_config.get(
            "allowed_operations", ["SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP"]
        )

        # Query limits
        self._max_query_length = self._sql_config.get("max_query_length", 10000)
        self._query_timeout = self._sql_config.get("query_timeout", 30)
        self._max_result_rows = self._sql_config.get("max_result_rows", 1000)

        # Add parameters
        self._setup_parameters()

    def _setup_parameters(self) -> None:
        """Set up SQL executor parameters."""
        # Database type parameter
        self.add_parameter(
            Parameter(
                name="database_type",
                type=ParameterType.STRING,
                description="Type of database to connect to",
                required=True,
                enum_values=self._supported_databases,
            )
        )

        # Connection parameters
        self.add_parameter(
            Parameter(
                name="host",
                type=ParameterType.STRING,
                description="Database host (default: localhost)",
                required=False,
                default="localhost",
            )
        )

        self.add_parameter(
            Parameter(
                name="port",
                type=ParameterType.INTEGER,
                description="Database port",
                required=False,
                min_value=1,
                max_value=65535,
            )
        )

        self.add_parameter(
            Parameter(
                name="database",
                type=ParameterType.STRING,
                description="Database name",
                required=True,
            )
        )

        self.add_parameter(
            Parameter(
                name="username",
                type=ParameterType.STRING,
                description="Database username",
                required=False,
            )
        )

        self.add_parameter(
            Parameter(
                name="password",
                type=ParameterType.STRING,
                description="Database password",
                required=False,
            )
        )

        # SQL query parameter
        self.add_parameter(
            Parameter(
                name="query",
                type=ParameterType.STRING,
                description="SQL query to execute",
                required=True,
            )
        )

        # Query parameters for prepared statements
        self.add_parameter(
            Parameter(
                name="parameters",
                type=ParameterType.ARRAY,
                description="Parameters for prepared statements",
                required=False,
                default=[],
            )
        )

        # Output format parameter
        self.add_parameter(
            Parameter(
                name="output_format",
                type=ParameterType.STRING,
                description="Format for query results",
                required=False,
                default="json",
                enum_values=["json", "csv", "table", "raw"],
            )
        )

        # Transaction parameter
        self.add_parameter(
            Parameter(
                name="use_transaction",
                type=ParameterType.BOOLEAN,
                description="Whether to wrap query in a transaction",
                required=False,
                default=False,
            )
        )

        # Read-only mode parameter
        self.add_parameter(
            Parameter(
                name="read_only",
                type=ParameterType.BOOLEAN,
                description="Restrict to read-only operations (SELECT only)",
                required=False,
                default=False,
            )
        )

    def validate_sql_query(self, query: str, read_only: bool = False) -> ValidationResult:
        """
        Validate SQL query for security and format.

        Args:
            query: SQL query string to validate
            read_only: Whether to restrict to read-only operations

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult.success()

        if not query or not query.strip():
            result.add_error("SQL query cannot be empty")
            return result

        # Check query length
        if len(query) > self._max_query_length:
            result.add_error(f"Query exceeds maximum length of {self._max_query_length} characters")

        # Normalize query for analysis
        normalized_query = query.lower().strip()

        # Check for dangerous patterns
        for pattern in self._dangerous_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                result.add_error(f"Query contains potentially dangerous pattern: {pattern}")

        # Check if read-only mode is enforced
        if read_only:
            if not normalized_query.startswith("select"):
                result.add_error("Only SELECT queries are allowed in read-only mode")

        # Check for allowed operations
        first_word = normalized_query.split()[0] if normalized_query.split() else ""
        if first_word.upper() not in self._allowed_operations:
            result.add_warning(f"Operation '{first_word.upper()}' may not be allowed")

        # Basic SQL injection checks
        injection_patterns = [
            r"(\'\s*or\s*\'|\"\s*or\s*\")",  # OR injection
            r"(\'\s*union\s*|\"\s*union\s*)",  # UNION injection
            r";\s*(drop|delete|insert|update)",  # Stacked queries
            r"\/\*.*\*\/",  # Block comments
            r"--[^\n]*",  # Line comments
        ]

        for pattern in injection_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                result.add_warning(f"Query contains pattern that might indicate SQL injection: {pattern}")

        return result

    def get_database_client_command(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Build database client command based on database type.

        Args:
            parameters: Database connection parameters

        Returns:
            List of command arguments for database client
        """
        db_type = parameters["database_type"].lower()
        host = parameters.get("host", "localhost")
        port = parameters.get("port")
        database = parameters["database"]
        username = parameters.get("username")
        password = parameters.get("password")

        if db_type == "postgresql":
            cmd = ["psql"]
            if port:
                cmd.extend(["-p", str(port)])
            cmd.extend(["-h", host, "-d", database])
            if username:
                cmd.extend(["-U", username])
            cmd.extend(["-t", "-A", "-F", ","])  # Tuples only, aligned, comma separator

        elif db_type == "mysql" or db_type == "mariadb":
            cmd = ["mysql"]
            if port:
                cmd.extend(["-P", str(port)])
            cmd.extend(["-h", host, database])
            if username:
                cmd.extend(["-u", username])
            if password:
                cmd.extend([f"-p{password}"])
            cmd.extend(["--batch", "--raw"])  # Batch mode, no formatting

        elif db_type == "sqlite":
            # For SQLite, the database parameter is the file path
            cmd = ["sqlite3", database, "-header", "-csv"]

        else:
            raise SandboxExecutionError(f"Unsupported database type: {db_type}")

        return cmd

    def build_sql_script(self, parameters: Dict[str, Any]) -> str:
        """
        Build SQL script with proper formatting and safety measures.

        Args:
            parameters: Execution parameters

        Returns:
            Complete SQL script as string
        """
        query = parameters["query"].strip()
        use_transaction = parameters.get("use_transaction", False)
        db_type = parameters["database_type"].lower()

        script_lines = []

        # Add timeout setting
        if db_type == "postgresql":
            script_lines.append("\\timing on")
            script_lines.append(f"SET statement_timeout = '{self._query_timeout}s';")
        elif db_type in ["mysql", "mariadb"]:
            script_lines.append(f"SET SESSION max_execution_time = {self._query_timeout * 1000};")

        # Add transaction wrapper if requested
        if use_transaction:
            script_lines.append("BEGIN;")

        # Add the main query
        if not query.endswith(";"):
            query += ";"
        script_lines.append(query)

        # Close transaction if needed
        if use_transaction:
            script_lines.append("COMMIT;")

        # Add exit command
        if db_type == "postgresql":
            script_lines.append("\\q")
        elif db_type in ["mysql", "mariadb"]:
            script_lines.append("exit;")
        elif db_type == "sqlite":
            script_lines.append(".quit")

        return "\n".join(script_lines)

    def parse_sql_output(
        self, stdout: str, stderr: str, return_code: int, query: str, output_format: str
    ) -> CommandResult:
        """
        Parse SQL output into a structured result.

        Args:
            stdout: Standard output from SQL client
            stderr: Standard error from SQL client
            return_code: Process exit code
            query: Original SQL query
            output_format: Requested output format

        Returns:
            CommandResult with structured query results
        """
        success = return_code == 0

        result_data = {
            "query": query,
            "success": success,
            "return_code": return_code,
            "output_format": output_format,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
        }

        if success and stdout.strip():
            try:
                lines = stdout.strip().split("\n")

                if output_format == "json":
                    # Parse CSV-like output to JSON
                    if lines:
                        # Try to identify header row
                        data_lines = [line for line in lines if line.strip() and not line.startswith("(")]

                        if data_lines:
                            # Assume first line might be headers or data
                            if len(data_lines) > 1:
                                headers = data_lines[0].split(",")
                                rows = []
                                for line in data_lines[1:]:
                                    values = line.split(",")
                                    if len(values) == len(headers):
                                        row = dict(zip(headers, values))
                                        rows.append(row)
                                result_data["results"] = rows
                            else:
                                # Single row result
                                result_data["results"] = [{"result": data_lines[0]}]

                elif output_format == "csv":
                    result_data["results"] = stdout.strip()

                elif output_format == "table":
                    result_data["results"] = stdout.strip()

                else:  # raw
                    result_data["results"] = stdout.strip()

                # Extract row count if available
                row_count_match = re.search(r"\((\d+) rows?\)", stdout)
                if row_count_match:
                    result_data["row_count"] = int(row_count_match.group(1))

            except Exception as e:
                result_data["parse_error"] = str(e)
                result_data["results"] = stdout.strip()

        metadata = {
            "command_type": "sql_query",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            error_msg = f"SQL query failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute SQL query in Docker container.

        Args:
            parameters: Execution parameters including query and connection details
            context: Execution context including session_id

        Returns:
            CommandResult with query results
        """
        try:
            # Extract session ID
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for SQL execution")

            # Get Docker environment
            environment = self.get_session_environment(session_id)

            # Validate SQL query
            read_only = parameters.get("read_only", False)
            query_validation = self.validate_sql_query(parameters["query"], read_only)
            if not query_validation.valid:
                return CommandResult.error_result(
                    error=f"SQL query validation failed: {', '.join(query_validation.errors)}"
                )

            # Log any warnings
            if query_validation.warnings:
                logger.warning(f"SQL query warnings: {', '.join(query_validation.warnings)}")

            # Build SQL script
            sql_script = self.build_sql_script(parameters)

            # Create script file in container
            script_path = f"/tmp/query_{session_id}.sql"
            create_script_cmd = ["sh", "-c", f"cat > {script_path} << 'EOF'\n{sql_script}\nEOF"]
            create_result = environment.execute_command(command=create_script_cmd)

            if create_result.exit_code != 0:
                return CommandResult.error_result(error=f"Failed to create SQL script: {create_result.stderr}")

            # Build database client command
            try:
                db_cmd = self.get_database_client_command(parameters)
                # Add script input
                full_cmd = (
                    db_cmd + ["-f", script_path]
                    if parameters["database_type"] == "postgresql"
                    else db_cmd + ["<", script_path]
                )

                # For non-PostgreSQL, use shell to handle input redirection
                if parameters["database_type"] != "postgresql":
                    full_cmd = ["sh", "-c", " ".join(db_cmd) + f" < {script_path}"]

            except Exception as e:
                return CommandResult.error_result(error=f"Failed to build database command: {e}")

            # Set environment variables for database connection
            env_vars = {}
            if parameters.get("password"):
                if parameters["database_type"] == "postgresql":
                    env_vars["PGPASSWORD"] = parameters["password"]
                elif parameters["database_type"] in ["mysql", "mariadb"]:
                    env_vars["MYSQL_PWD"] = parameters["password"]

            # Prepend environment variables to command if any
            if env_vars:
                env_cmd = []
                for key, value in env_vars.items():
                    env_cmd.extend(["env", f"{key}={value}"])
                full_cmd = env_cmd + full_cmd

            # Execute SQL command
            result = environment.execute_command(command=full_cmd)

            # Parse output
            output_format = parameters.get("output_format", "json")
            tool_result = self.parse_sql_output(
                result.stdout, result.stderr, result.exit_code, parameters["query"], output_format
            )

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "session_id": session_id,
                    "execution_time": result.execution_time,
                    "database_type": parameters["database_type"],
                    "query_length": len(parameters["query"]),
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return CommandResult.error_result(f"SQL execution failed: {str(e)}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for SQL execution.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with comprehensive validation
        """
        # Run base Docker validation
        result = super().validate_parameters(parameters)

        # Add SQL-specific validation
        if "query" in parameters:
            read_only = parameters.get("read_only", False)
            query_validation = self.validate_sql_query(parameters["query"], read_only)
            result.errors.extend(query_validation.errors)
            result.warnings.extend(query_validation.warnings)

        # Validate database type
        db_type = parameters.get("database_type")
        if db_type and db_type not in self._supported_databases:
            result.add_error(f"Unsupported database type: {db_type}")

        # Validate connection parameters
        if "port" in parameters:
            port = parameters["port"]
            if not isinstance(port, int) or port < 1 or port > 65535:
                result.add_error("Port must be an integer between 1 and 65535")

        # Validate parameters array
        params = parameters.get("parameters", [])
        if params and not isinstance(params, list):
            result.add_error("parameters must be a list")

        return result
