"""
Docker-based curl executor for HTTP/HTTPS requests in isolated containers.

This module provides a secure curl executor that accepts HTTP request parameters and executes
curl commands in Docker containers with proper security validation and response handling.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class CurlExecutor(DockerExecutor):
    """
    Docker-based curl executor for secure HTTP/HTTPS requests.

    This executor provides HTTP request capabilities including:
    - URL validation and security checking
    - HTTP method support (GET, POST, PUT, DELETE, etc.)
    - Header and authentication management
    - Request body handling (JSON, form data, raw)
    - Response parsing and formatting
    - SSL/TLS certificate handling
    """

    _security_command_metadata = {
        "domain": "network",
        "name": "curl_request",
        "description": "Execute HTTP/HTTPS requests using curl in Docker containers",
        "author": "SABER Team",
        "security_level": "medium",
        "requires_validation": True,
    }

    def __init__(
        self, sandbox_manager: SandboxManager, curl_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize curl executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            curl_config: Optional curl-specific configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, docker_config=curl_config, **kwargs)

        self._curl_config = curl_config or {}

        # Set up allowed protocols and domains (security feature)
        self._allowed_protocols = self._curl_config.get("allowed_protocols", ["http", "https"])
        self._blocked_domains = self._curl_config.get(
            "blocked_domains", ["localhost", "127.0.0.1", "0.0.0.0", "::1", "169.254.169.254"]
        )
        self._allowed_domains = self._curl_config.get("allowed_domains", [])  # Empty means all allowed
        self._max_response_size = self._curl_config.get("max_response_size", 10 * 1024 * 1024)  # 10MB
        self._timeout = self._curl_config.get("timeout", 30)  # 30 seconds

        # Add parameters
        self._setup_parameters()

    def _setup_parameters(self) -> None:
        """Set up curl executor parameters."""
        # URL parameter
        self.add_parameter(
            Parameter(
                name="url",
                type=ParameterType.STRING,
                description="Target URL for the HTTP request",
                required=True,
            )
        )

        # HTTP method parameter
        self.add_parameter(
            Parameter(
                name="method",
                type=ParameterType.STRING,
                description="HTTP method to use",
                required=False,
                default="GET",
                enum_values=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"],
            )
        )

        # Headers parameter
        self.add_parameter(
            Parameter(
                name="headers",
                type=ParameterType.OBJECT,
                description="HTTP headers as key-value pairs",
                required=False,
                default={},
            )
        )

        # Request body parameter
        self.add_parameter(
            Parameter(
                name="data",
                type=ParameterType.STRING,
                description="Request body data (JSON string, form data, or raw text)",
                required=False,
            )
        )

        # Data type parameter
        self.add_parameter(
            Parameter(
                name="data_type",
                type=ParameterType.STRING,
                description="Type of data being sent",
                required=False,
                default="raw",
                enum_values=["json", "form", "raw"],
            )
        )

        # Authentication parameter
        self.add_parameter(
            Parameter(
                name="auth",
                type=ParameterType.OBJECT,
                description=(
                    "Authentication credentials \
                    (e.g., {'type': 'basic', 'username': 'user', 'password': 'pass'})"
                ),
                required=False,
            )
        )

        # SSL verification parameter
        self.add_parameter(
            Parameter(
                name="verify_ssl",
                type=ParameterType.BOOLEAN,
                description="Whether to verify SSL certificates",
                required=False,
                default=True,
            )
        )

        # Follow redirects parameter
        self.add_parameter(
            Parameter(
                name="follow_redirects",
                type=ParameterType.BOOLEAN,
                description="Whether to follow HTTP redirects",
                required=False,
                default=True,
            )
        )

        # Output format parameter
        self.add_parameter(
            Parameter(
                name="output_format",
                type=ParameterType.STRING,
                description="Format for response output",
                required=False,
                default="full",
                enum_values=["full", "headers_only", "body_only", "status_only"],
            )
        )

    def validate_url(self, url: str) -> ValidationResult:
        """
        Validate URL for security and format.

        Args:
            url: URL string to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult.success()

        if not url or not url.strip():
            result.add_error("URL cannot be empty")
            return result

        # Basic URL format validation
        url_pattern = re.compile(
            r"^(https?):\/\/"  # Protocol
            r"([A-Za-z0-9\-._~:/?#[\]@!$&\'()*+,;=%]+)"  # Rest of URL
        )

        if not url_pattern.match(url):
            result.add_error("Invalid URL format")
            return result

        # Extract components
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)

            # Check protocol
            if parsed.scheme not in self._allowed_protocols:
                result.add_error(f"Protocol '{parsed.scheme}' not allowed. Allowed: {self._allowed_protocols}")

            # Check for blocked domains
            hostname = parsed.hostname or ""
            for blocked in self._blocked_domains:
                if blocked in hostname:
                    result.add_error(f"Domain '{hostname}' is blocked")

            # Check allowed domains (if specified)
            if self._allowed_domains:
                allowed = any(allowed_domain in hostname for allowed_domain in self._allowed_domains)
                if not allowed:
                    result.add_error(f"Domain '{hostname}' is not in allowed list")

        except Exception as e:
            result.add_error(f"URL parsing error: {e}")

        return result

    def build_curl_command(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Build curl command from parameters.

        Args:
            parameters: Execution parameters

        Returns:
            List of command arguments for curl
        """
        cmd = ["curl"]

        # Add basic options
        cmd.extend(["-s", "-S"])  # Silent but show errors
        cmd.extend(["--max-time", str(self._timeout)])
        cmd.extend(["--max-filesize", str(self._max_response_size)])

        # Add method
        method = parameters.get("method", "GET")
        if method != "GET":
            cmd.extend(["-X", method])

        # Add headers
        headers = parameters.get("headers", {})
        for key, value in headers.items():
            cmd.extend(["-H", f"{key}: {value}"])

        # Add authentication
        auth = parameters.get("auth")
        if auth:
            auth_type = auth.get("type", "").lower()
            if auth_type == "basic":
                username = auth.get("username", "")
                password = auth.get("password", "")
                cmd.extend(["-u", f"{username}:{password}"])
            elif auth_type == "bearer":
                token = auth.get("token", "")
                cmd.extend(["-H", f"Authorization: Bearer {token}"])

        # Add data
        data = parameters.get("data")
        data_type = parameters.get("data_type", "raw")
        if data:
            if data_type == "json":
                cmd.extend(["-H", "Content-Type: application/json"])
                cmd.extend(["-d", data])
            elif data_type == "form":
                cmd.extend(["-H", "Content-Type: application/x-www-form-urlencoded"])
                cmd.extend(["-d", data])
            else:  # raw
                cmd.extend(["-d", data])

        # SSL verification
        if not parameters.get("verify_ssl", True):
            cmd.append("-k")

        # Follow redirects
        if parameters.get("follow_redirects", True):
            cmd.extend(["-L", "--max-redirs", "5"])

        # Output format
        output_format = parameters.get("output_format", "full")
        if output_format == "headers_only":
            cmd.append("-I")
        elif output_format == "body_only":
            cmd.extend(["-s", "-o", "-"])
        elif output_format == "status_only":
            cmd.extend(["-w", "%{http_code}", "-o", "/dev/null"])
        else:  # full
            cmd.extend(["-i"])  # Include headers with body

        # Add URL
        cmd.append(parameters["url"])

        return cmd

    def parse_curl_output(self, stdout: str, stderr: str, return_code: int, url: str) -> CommandResult:
        """
        Parse curl output into a structured result.

        Args:
            stdout: Standard output from curl
            stderr: Standard error from curl
            return_code: Process exit code
            url: Target URL

        Returns:
            CommandResult with structured HTTP response data
        """
        success = return_code == 0

        result_data = {
            "url": url,
            "success": success,
            "return_code": return_code,
            "raw_stdout": stdout,
            "raw_stderr": stderr,
        }

        if success and stdout:
            # Parse HTTP response
            try:
                # Split headers and body (if both present)
                if "\r\n\r\n" in stdout:
                    headers_part, body_part = stdout.split("\r\n\r\n", 1)
                elif "\n\n" in stdout:
                    headers_part, body_part = stdout.split("\n\n", 1)
                else:
                    headers_part = stdout
                    body_part = ""

                # Parse status line and headers
                lines = headers_part.split("\n")
                status_line = lines[0] if lines else ""

                # Extract status code
                status_code_match = re.search(r"HTTP/[\d.]+ (\d+)", status_line)
                status_code = int(status_code_match.group(1)) if status_code_match else 0

                # Parse headers
                headers = {}
                for line in lines[1:]:
                    if ":" in line:
                        key, value = line.split(":", 1)
                        headers[key.strip()] = value.strip()

                result_data.update(
                    {
                        "status_code": status_code,
                        "status_line": status_line.strip(),
                        "headers": headers,
                        "body": body_part,
                        "response_size": len(body_part),
                    }
                )

                # Try to parse JSON body
                if body_part.strip():
                    content_type = headers.get("Content-Type", "").lower()
                    if "application/json" in content_type:
                        try:
                            result_data["json"] = json.loads(body_part)
                        except json.JSONDecodeError:
                            pass  # Not valid JSON, keep as text

            except Exception as e:
                result_data["parse_error"] = str(e)

        metadata = {
            "command_type": "curl_request",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            error_msg = f"Curl request failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute curl request in Docker container.

        Args:
            parameters: Execution parameters including URL and request options
            context: Execution context including session_id

        Returns:
            CommandResult with HTTP response results
        """
        try:
            # Extract session ID
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for curl execution")

            # Get Docker environment
            environment = self.get_session_environment(session_id)

            # Validate URL
            url_validation = self.validate_url(parameters["url"])
            if not url_validation.valid:
                return CommandResult.error_result(error=f"URL validation failed: {', '.join(url_validation.errors)}")

            # Log any warnings
            if url_validation.warnings:
                logger.warning(f"URL warnings: {', '.join(url_validation.warnings)}")

            # Build curl command
            curl_cmd = self.build_curl_command(parameters)

            # Execute curl command
            result = environment.execute_command(command=curl_cmd)

            # Parse output
            tool_result = self.parse_curl_output(result.stdout, result.stderr, result.exit_code, parameters["url"])

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "session_id": session_id,
                    "execution_time": result.execution_time,
                    "curl_command": " ".join(curl_cmd),
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"Curl execution error: {e}")
            return CommandResult.error_result(f"Curl execution failed: {str(e)}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for curl execution.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with comprehensive validation
        """
        # Run base Docker validation
        result = super().validate_parameters(parameters)

        # Add curl-specific validation
        if "url" in parameters:
            url_validation = self.validate_url(parameters["url"])
            result.errors.extend(url_validation.errors)
            result.warnings.extend(url_validation.warnings)

        # Validate headers
        headers = parameters.get("headers", {})
        if headers and not isinstance(headers, dict):
            result.add_error("headers must be a dictionary")

        # Validate auth
        auth = parameters.get("auth")
        if auth:
            if not isinstance(auth, dict):
                result.add_error("auth must be a dictionary")
            else:
                auth_type = auth.get("type", "").lower()
                if auth_type == "basic":
                    if "username" not in auth or "password" not in auth:
                        result.add_error("basic auth requires username and password")
                elif auth_type == "bearer":
                    if "token" not in auth:
                        result.add_error("bearer auth requires token")

        return result
