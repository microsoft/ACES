"""
Tests for Curl executor.

This module tests the secure curl executor that executes HTTP/HTTPS requests
in Docker containers with security validation and response handling.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.execution.base import CommandResult, ParameterType, ValidationResult
from saber.server.execution.executors.curl_executor import CurlExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class TestCurlExecutor:
    """Test cases for Curl executor."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/curl-sandbox:latest",
            "network_mode": "bridge",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def curl_executor(self, mock_sandbox_manager):
        """Create a Curl executor instance for testing."""
        return CurlExecutor(sandbox_manager=mock_sandbox_manager, timeout=60.0)

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
        """Test curl executor initialization."""
        executor = CurlExecutor(sandbox_manager=mock_sandbox_manager)

        assert executor._sandbox_manager == mock_sandbox_manager
        assert executor._allowed_protocols == ["http", "https"]
        assert "localhost" in executor._blocked_domains
        assert executor._max_response_size == 10 * 1024 * 1024
        assert executor._timeout == 30

    def test_initialization_with_config(self, mock_sandbox_manager):
        """Test curl executor initialization with custom config."""
        config = {
            "allowed_protocols": ["https"],
            "blocked_domains": ["example.com"],
            "timeout": 60,
            "max_response_size": 5 * 1024 * 1024
        }
        executor = CurlExecutor(sandbox_manager=mock_sandbox_manager, curl_config=config)

        assert executor._allowed_protocols == ["https"]
        assert executor._blocked_domains == ["example.com"]
        assert executor._timeout == 60
        assert executor._max_response_size == 5 * 1024 * 1024

    def test_parameters_setup(self, curl_executor):
        """Test that curl executor sets up required parameters."""
        params = curl_executor.get_parameters()

        assert "url" in params
        assert params["url"].required is True
        assert params["url"].type == ParameterType.STRING

        assert "method" in params
        assert params["method"].default == "GET"
        assert "GET" in params["method"].enum_values
        assert "POST" in params["method"].enum_values

        assert "headers" in params
        assert params["headers"].type == ParameterType.OBJECT
        assert params["headers"].required is False

    def test_validate_url_valid(self, curl_executor):
        """Test URL validation with valid URLs."""
        valid_urls = [
            "https://example.com",
            "http://api.example.com/v1/data",
            "https://subdomain.example.com:8080/path?query=value",
        ]

        for url in valid_urls:
            result = curl_executor.validate_url(url)
            assert result.valid, f"URL should be valid: {url}"

    def test_validate_url_invalid_format(self, curl_executor):
        """Test URL validation with invalid format."""
        invalid_urls = [
            "",
            "not-a-url",
            "ftp://example.com",
            "javascript:alert('xss')",
        ]

        for url in invalid_urls:
            result = curl_executor.validate_url(url)
            assert not result.valid, f"URL should be invalid: {url}"

    def test_validate_url_blocked_domains(self, curl_executor):
        """Test URL validation with blocked domains."""
        blocked_urls = [
            "http://localhost:8080",
            "https://127.0.0.1/admin",
            "http://169.254.169.254/metadata",
        ]

        for url in blocked_urls:
            result = curl_executor.validate_url(url)
            assert not result.valid, f"URL should be blocked: {url}"

    def test_validate_url_protocol_restriction(self, mock_sandbox_manager):
        """Test URL validation with protocol restrictions."""
        config = {"allowed_protocols": ["https"]}
        executor = CurlExecutor(sandbox_manager=mock_sandbox_manager, curl_config=config)

        # HTTPS should be allowed
        result = executor.validate_url("https://example.com")
        assert result.valid

        # HTTP should be blocked
        result = executor.validate_url("http://example.com")
        assert not result.valid

    def test_build_curl_command_get(self, curl_executor):
        """Test building curl command for GET request."""
        parameters = {
            "url": "https://api.example.com/data",
            "method": "GET"
        }

        cmd = curl_executor.build_curl_command(parameters)

        assert "curl" in cmd
        assert "-s" in cmd
        assert "-S" in cmd
        assert "https://api.example.com/data" in cmd
        assert "-X" not in cmd  # GET is default

    def test_build_curl_command_post_with_json(self, curl_executor):
        """Test building curl command for POST with JSON data."""
        parameters = {
            "url": "https://api.example.com/submit",
            "method": "POST",
            "data": '{"key": "value"}',
            "data_type": "json",
            "headers": {"Authorization": "Bearer token123"}
        }

        cmd = curl_executor.build_curl_command(parameters)

        assert "curl" in cmd
        assert "-X" in cmd
        assert "POST" in cmd
        assert "-d" in cmd
        assert '{"key": "value"}' in cmd
        assert "-H" in cmd
        assert any("Content-Type: application/json" in str(item) for item in cmd)
        assert any("Authorization: Bearer token123" in str(item) for item in cmd)

    def test_build_curl_command_with_auth(self, curl_executor):
        """Test building curl command with authentication."""
        parameters = {
            "url": "https://api.example.com/secure",
            "auth": {
                "type": "basic",
                "username": "user",
                "password": "pass"
            }
        }

        cmd = curl_executor.build_curl_command(parameters)

        assert "-u" in cmd
        assert "user:pass" in cmd

    def test_build_curl_command_bearer_auth(self, curl_executor):
        """Test building curl command with bearer token."""
        parameters = {
            "url": "https://api.example.com/secure",
            "auth": {
                "type": "bearer",
                "token": "abc123"
            }
        }

        cmd = curl_executor.build_curl_command(parameters)

        assert "-H" in cmd
        assert any("Authorization: Bearer abc123" in str(item) for item in cmd)

    def test_build_curl_command_ssl_options(self, curl_executor):
        """Test building curl command with SSL options."""
        parameters = {
            "url": "https://api.example.com",
            "verify_ssl": False,
            "follow_redirects": True
        }

        cmd = curl_executor.build_curl_command(parameters)

        assert "-k" in cmd  # Skip SSL verification
        assert "-L" in cmd  # Follow redirects

    def test_parse_curl_output_success(self, curl_executor):
        """Test parsing successful curl output."""
        stdout = """HTTP/1.1 200 OK
Content-Type: application/json
Content-Length: 13

{"status":"ok"}"""
        stderr = ""
        return_code = 0
        url = "https://api.example.com"

        result = curl_executor.parse_curl_output(stdout, stderr, return_code, url)

        assert result.success
        assert result.data["status_code"] == 200
        assert result.data["headers"]["Content-Type"] == "application/json"
        assert result.data["body"] == '{"status":"ok"}'
        assert "json" in result.data
        assert result.data["json"]["status"] == "ok"

    def test_parse_curl_output_error(self, curl_executor):
        """Test parsing curl output with error."""
        stdout = ""
        stderr = "curl: (6) Could not resolve host: nonexistent.example.com"
        return_code = 6
        url = "https://nonexistent.example.com"

        result = curl_executor.parse_curl_output(stdout, stderr, return_code, url)

        assert not result.success
        assert "Could not resolve host" in result.error

    def test_parse_curl_output_headers_only(self, curl_executor):
        """Test parsing curl output with headers only."""
        stdout = "HTTP/1.1 200 OK\nContent-Type: text/html\n"
        stderr = ""
        return_code = 0
        url = "https://example.com"

        result = curl_executor.parse_curl_output(stdout, stderr, return_code, url)

        assert result.success
        assert result.data["status_code"] == 200
        assert result.data["headers"]["Content-Type"] == "text/html"

    @pytest.mark.asyncio
    async def test_execute_success(self, curl_executor, mock_docker_environment):
        """Test successful curl execution."""
        # Setup
        parameters = {
            "url": "https://httpbin.org/get",
            "method": "GET"
        }
        context = {"session_id": "test-session"}

        # Mock environment
        curl_executor.get_session_environment = MagicMock(return_value=mock_docker_environment)

        # Mock command execution
        mock_result = MagicMock()
        mock_result.stdout = "HTTP/1.1 200 OK\nContent-Type: application/json\n\n{\"url\":\"https://httpbin.org/get\"}"
        mock_result.stderr = ""
        mock_result.exit_code = 0
        mock_result.execution_time = 1.5
        mock_docker_environment.execute_command.return_value = mock_result

        # Execute
        result = await curl_executor.execute(parameters, context)

        # Verify
        assert result.success
        assert result.data["url"] == "https://httpbin.org/get"
        assert result.data["status_code"] == 200
        assert result.metadata["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_execute_url_validation_failure(self, curl_executor):
        """Test curl execution with URL validation failure."""
        parameters = {
            "url": "http://localhost:8080/admin",
            "method": "GET"
        }
        context = {"session_id": "test-session"}

        result = await curl_executor.execute(parameters, context)

        assert not result.success
        assert "URL validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, curl_executor):
        """Test curl execution without session_id."""
        parameters = {"url": "https://example.com"}
        context = {}

        result = await curl_executor.execute(parameters, context)

        assert not result.success
        assert "session_id required" in result.error

    def test_validate_parameters_success(self, curl_executor):
        """Test parameter validation with valid parameters."""
        parameters = {
            "url": "https://api.example.com",
            "method": "POST",
            "headers": {"Content-Type": "application/json"},
            "auth": {"type": "basic", "username": "user", "password": "pass"}
        }

        result = curl_executor.validate_parameters(parameters)
        assert result.valid

    def test_validate_parameters_invalid_headers(self, curl_executor):
        """Test parameter validation with invalid headers."""
        parameters = {
            "url": "https://api.example.com",
            "headers": "invalid-headers"  # Should be dict
        }

        result = curl_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("headers must be a dictionary" in error for error in result.errors)

    def test_validate_parameters_invalid_auth(self, curl_executor):
        """Test parameter validation with invalid auth."""
        parameters = {
            "url": "https://api.example.com",
            "auth": {"type": "basic", "username": "user"}  # Missing password
        }

        result = curl_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("basic auth requires username and password" in error for error in result.errors)

    def test_validate_parameters_bearer_auth(self, curl_executor):
        """Test parameter validation with bearer auth."""
        parameters = {
            "url": "https://api.example.com",
            "auth": {"type": "bearer"}  # Missing token
        }

        result = curl_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("bearer auth requires token" in error for error in result.errors)

    def test_to_mcp_schema(self, curl_executor):
        """Test MCP schema generation."""
        schema = curl_executor.to_mcp_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "url" in schema["properties"]
        assert "method" in schema["properties"]
        assert "required" in schema
        assert "url" in schema["required"]

    def test_security_command_metadata(self, curl_executor):
        """Test security command metadata."""
        metadata = curl_executor._security_command_metadata

        assert metadata["domain"] == "network"
        assert metadata["name"] == "curl_request"
        assert metadata["security_level"] == "medium"
        assert metadata["requires_validation"] is True
