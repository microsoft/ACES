"""
Tests for SQL executor.

This module tests the secure SQL executor that executes database queries
in Docker containers with security validation and result formatting.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.execution.base import CommandResult, ParameterType, ValidationResult
from saber.server.execution.executors.sql_executor import SqlExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class TestSqlExecutor:
    """Test cases for SQL executor."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/sql-sandbox:latest",
            "network_mode": "bridge",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def sql_executor(self, mock_sandbox_manager):
        """Create a SQL executor instance for testing."""
        return SqlExecutor(sandbox_manager=mock_sandbox_manager, timeout=60.0)

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        env = MagicMock()
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = MagicMock()
        return env

    def test_initialization(self, mock_sandbox_manager):
        """Test SQL executor initialization."""
        executor = SqlExecutor(sandbox_manager=mock_sandbox_manager)

        assert executor._sandbox_manager == mock_sandbox_manager
        assert "postgresql" in executor._supported_databases
        assert "mysql" in executor._supported_databases
        assert "sqlite" in executor._supported_databases
        assert executor._max_query_length == 10000
        assert executor._query_timeout == 30

    def test_initialization_with_config(self, mock_sandbox_manager):
        """Test SQL executor initialization with custom config."""
        config = {
            "supported_databases": ["postgresql"],
            "max_query_length": 5000,
            "query_timeout": 60,
            "max_result_rows": 500
        }
        executor = SqlExecutor(sandbox_manager=mock_sandbox_manager, sql_config=config)

        assert executor._supported_databases == ["postgresql"]
        assert executor._max_query_length == 5000
        assert executor._query_timeout == 60
        assert executor._max_result_rows == 500

    def test_parameters_setup(self, sql_executor):
        """Test that SQL executor sets up required parameters."""
        params = sql_executor.get_parameters()

        assert "database_type" in params
        assert params["database_type"].required is True
        assert "postgresql" in params["database_type"].enum_values

        assert "query" in params
        assert params["query"].required is True
        assert params["query"].type == ParameterType.STRING

        assert "database" in params
        assert params["database"].required is True

        assert "host" in params
        assert params["host"].default == "localhost"

    def test_validate_sql_query_valid(self, sql_executor):
        """Test SQL query validation with valid queries."""
        valid_queries = [
            "SELECT * FROM users WHERE id = 1",
            "INSERT INTO logs (message) VALUES ('test')",
            "UPDATE users SET name = 'John' WHERE id = 1",
            "DELETE FROM temp_data WHERE created < '2023-01-01'",
        ]

        for query in valid_queries:
            result = sql_executor.validate_sql_query(query)
            assert result.valid, f"Query should be valid: {query}"

    def test_validate_sql_query_dangerous_patterns(self, sql_executor):
        """Test SQL query validation with dangerous patterns."""
        dangerous_queries = [
            "DROP DATABASE production",
            "DROP TABLE users",
            "DELETE FROM users WHERE 1=1",
            "TRUNCATE TABLE logs",
            "GRANT ALL ON *.* TO 'user'@'%'",
            "SHUTDOWN",
        ]

        for query in dangerous_queries:
            result = sql_executor.validate_sql_query(query)
            assert not result.valid, f"Query should be dangerous: {query}"

    def test_validate_sql_query_read_only_mode(self, sql_executor):
        """Test SQL query validation in read-only mode."""
        # SELECT should be allowed
        result = sql_executor.validate_sql_query("SELECT * FROM users", read_only=True)
        assert result.valid

        # Non-SELECT should be blocked
        result = sql_executor.validate_sql_query("INSERT INTO users VALUES (1, 'test')", read_only=True)
        assert not result.valid

    def test_validate_sql_query_empty(self, sql_executor):
        """Test SQL query validation with empty query."""
        result = sql_executor.validate_sql_query("")
        assert not result.valid
        assert "cannot be empty" in result.errors[0]

    def test_validate_sql_query_too_long(self, sql_executor):
        """Test SQL query validation with overly long query."""
        long_query = "SELECT * FROM users WHERE " + " OR ".join([f"id = {i}" for i in range(1000)])
        result = sql_executor.validate_sql_query(long_query)
        assert not result.valid
        assert "exceeds maximum length" in result.errors[0]

    def test_get_database_client_command_postgresql(self, sql_executor):
        """Test building PostgreSQL client command."""
        parameters = {
            "database_type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "testuser"
        }

        cmd = sql_executor.get_database_client_command(parameters)

        assert "psql" in cmd
        assert "-h" in cmd
        assert "localhost" in cmd
        assert "-p" in cmd
        assert "5432" in cmd
        assert "-d" in cmd
        assert "testdb" in cmd
        assert "-U" in cmd
        assert "testuser" in cmd

    def test_get_database_client_command_mysql(self, sql_executor):
        """Test building MySQL client command."""
        parameters = {
            "database_type": "mysql",
            "host": "localhost",
            "port": 3306,
            "database": "testdb",
            "username": "testuser",
            "password": "testpass"
        }

        cmd = sql_executor.get_database_client_command(parameters)

        assert "mysql" in cmd
        assert "-h" in cmd
        assert "localhost" in cmd
        assert "-P" in cmd
        assert "3306" in cmd
        assert "testdb" in cmd
        assert "-u" in cmd
        assert "testuser" in cmd
        assert "-ptestpass" in cmd

    def test_get_database_client_command_sqlite(self, sql_executor):
        """Test building SQLite client command."""
        parameters = {
            "database_type": "sqlite",
            "database": "/tmp/test.db"
        }

        cmd = sql_executor.get_database_client_command(parameters)

        assert "sqlite3" in cmd
        assert "/tmp/test.db" in cmd
        assert "-header" in cmd
        assert "-csv" in cmd

    def test_get_database_client_command_unsupported(self, sql_executor):
        """Test building command for unsupported database."""
        parameters = {
            "database_type": "oracle",
            "database": "testdb"
        }

        with pytest.raises(SandboxExecutionError) as excinfo:
            sql_executor.get_database_client_command(parameters)

        assert "Unsupported database type" in str(excinfo.value)

    def test_build_sql_script_simple(self, sql_executor):
        """Test building simple SQL script."""
        parameters = {
            "query": "SELECT * FROM users",
            "database_type": "postgresql",
            "use_transaction": False
        }

        script = sql_executor.build_sql_script(parameters)

        assert "SELECT * FROM users;" in script
        assert "\\timing on" in script
        assert "\\q" in script
        assert "BEGIN;" not in script

    def test_build_sql_script_with_transaction(self, sql_executor):
        """Test building SQL script with transaction."""
        parameters = {
            "query": "UPDATE users SET active = true",
            "database_type": "postgresql",
            "use_transaction": True
        }

        script = sql_executor.build_sql_script(parameters)

        assert "BEGIN;" in script
        assert "UPDATE users SET active = true;" in script
        assert "COMMIT;" in script

    def test_build_sql_script_mysql(self, sql_executor):
        """Test building SQL script for MySQL."""
        parameters = {
            "query": "SELECT COUNT(*) FROM orders",
            "database_type": "mysql",
            "use_transaction": False
        }

        script = sql_executor.build_sql_script(parameters)

        assert "SELECT COUNT(*) FROM orders;" in script
        assert "SET SESSION max_execution_time" in script
        assert "exit;" in script

    def test_parse_sql_output_success_json(self, sql_executor):
        """Test parsing successful SQL output to JSON."""
        stdout = "id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com\n(2 rows)"
        stderr = ""
        return_code = 0
        query = "SELECT * FROM users"

        result = sql_executor.parse_sql_output(stdout, stderr, return_code, query, "json")

        assert result.success
        assert result.data["query"] == query
        assert result.data["row_count"] == 2
        assert "results" in result.data
        assert len(result.data["results"]) == 2
        assert result.data["results"][0]["name"] == "John"

    def test_parse_sql_output_error(self, sql_executor):
        """Test parsing SQL output with error."""
        stdout = ""
        stderr = "ERROR: relation 'nonexistent_table' does not exist"
        return_code = 1
        query = "SELECT * FROM nonexistent_table"

        result = sql_executor.parse_sql_output(stdout, stderr, return_code, query, "json")

        assert not result.success
        assert "does not exist" in result.error

    def test_parse_sql_output_csv(self, sql_executor):
        """Test parsing SQL output as CSV."""
        stdout = "id,name\n1,John\n2,Jane"
        stderr = ""
        return_code = 0
        query = "SELECT id, name FROM users"

        result = sql_executor.parse_sql_output(stdout, stderr, return_code, query, "csv")

        assert result.success
        assert result.data["results"] == stdout.strip()

    @pytest.mark.asyncio
    async def test_execute_success(self, sql_executor, mock_docker_environment):
        """Test successful SQL execution."""
        # Setup
        parameters = {
            "database_type": "postgresql",
            "host": "localhost",
            "database": "testdb",
            "username": "testuser",
            "query": "SELECT COUNT(*) FROM users"
        }
        context = {"session_id": "test-session"}

        # Mock environment
        sql_executor.get_session_environment = MagicMock(return_value=mock_docker_environment)

        # Mock script creation
        create_result = MagicMock()
        create_result.exit_code = 0

        # Mock query execution
        query_result = MagicMock()
        query_result.stdout = "count\n5\n(1 row)"
        query_result.stderr = ""
        query_result.exit_code = 0
        query_result.execution_time = 0.5

        mock_docker_environment.execute_command.side_effect = [create_result, query_result]

        # Execute
        result = await sql_executor.execute(parameters, context)

        # Verify
        assert result.success
        assert result.data["query"] == "SELECT COUNT(*) FROM users"
        assert result.metadata["session_id"] == "test-session"
        assert result.metadata["database_type"] == "postgresql"

    @pytest.mark.asyncio
    async def test_execute_query_validation_failure(self, sql_executor):
        """Test SQL execution with query validation failure."""
        parameters = {
            "database_type": "postgresql",
            "database": "testdb",
            "query": "DROP DATABASE production"
        }
        context = {"session_id": "test-session"}

        result = await sql_executor.execute(parameters, context)

        assert not result.success
        assert "SQL query validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, sql_executor):
        """Test SQL execution without session_id."""
        parameters = {
            "database_type": "postgresql",
            "database": "testdb",
            "query": "SELECT 1"
        }
        context = {}

        result = await sql_executor.execute(parameters, context)

        assert not result.success
        assert "session_id required" in result.error

    @pytest.mark.asyncio
    async def test_execute_script_creation_failure(self, sql_executor, mock_docker_environment):
        """Test SQL execution with script creation failure."""
        parameters = {
            "database_type": "postgresql",
            "database": "testdb",
            "query": "SELECT 1"
        }
        context = {"session_id": "test-session"}

        sql_executor.get_session_environment = MagicMock(return_value=mock_docker_environment)

        # Mock script creation failure
        create_result = MagicMock()
        create_result.exit_code = 1
        create_result.stderr = "Permission denied"
        mock_docker_environment.execute_command.return_value = create_result

        result = await sql_executor.execute(parameters, context)

        assert not result.success
        assert "Failed to create SQL script" in result.error

    def test_validate_parameters_success(self, sql_executor):
        """Test parameter validation with valid parameters."""
        parameters = {
            "database_type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "testdb",
            "username": "testuser",
            "query": "SELECT * FROM users",
            "parameters": ["value1", "value2"]
        }

        result = sql_executor.validate_parameters(parameters)
        assert result.valid

    def test_validate_parameters_invalid_database_type(self, sql_executor):
        """Test parameter validation with invalid database type."""
        parameters = {
            "database_type": "oracle",
            "database": "testdb",
            "query": "SELECT 1"
        }

        result = sql_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("Unsupported database type" in error for error in result.errors)

    def test_validate_parameters_invalid_port(self, sql_executor):
        """Test parameter validation with invalid port."""
        parameters = {
            "database_type": "postgresql",
            "database": "testdb",
            "port": 99999,  # Invalid port
            "query": "SELECT 1"
        }

        result = sql_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("Port must be an integer between 1 and 65535" in error for error in result.errors)

    def test_validate_parameters_invalid_parameters_type(self, sql_executor):
        """Test parameter validation with invalid parameters type."""
        parameters = {
            "database_type": "postgresql",
            "database": "testdb",
            "query": "SELECT 1",
            "parameters": "not-a-list"  # Should be list
        }

        result = sql_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("parameters must be a list" in error for error in result.errors)

    def test_to_mcp_schema(self, sql_executor):
        """Test MCP schema generation."""
        schema = sql_executor.to_mcp_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "database_type" in schema["properties"]
        assert "query" in schema["properties"]
        assert "required" in schema
        assert "database_type" in schema["required"]
        assert "query" in schema["required"]

    def test_security_command_metadata(self, sql_executor):
        """Test security command metadata."""
        metadata = sql_executor._security_command_metadata

        assert metadata["domain"] == "database"
        assert metadata["name"] == "sql_query"
        assert metadata["security_level"] == "high"
        assert metadata["requires_validation"] is True
