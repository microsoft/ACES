"""
Docker-based file I/O executor for secure file operations in isolated containers.

This module provides a secure file I/O executor that handles file and directory operations
in Docker containers with proper security validation and path restrictions.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


class FileIoExecutor(DockerExecutor):
    """
    Docker-based file I/O executor for secure file operations.

    This executor provides file system capabilities including:
    - File reading, writing, and manipulation
    - Directory operations (list, create, remove)
    - Path validation and security restrictions
    - File content filtering and validation
    - Archive operations (tar, zip)
    - File permissions and ownership management
    """

    _security_command_metadata = {
        "domain": "filesystem",
        "name": "file_io",
        "description": "Execute file and directory operations in Docker containers",
        "author": "SABER Team",
        "security_level": "medium",
        "requires_validation": True,
    }

    def __init__(
        self, sandbox_manager: SandboxManager, fileio_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize file I/O executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            fileio_config: Optional file I/O-specific configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, docker_config=fileio_config, **kwargs)

        self._fileio_config = fileio_config or {}

        # Set up path restrictions (security feature)
        self._allowed_base_paths = self._fileio_config.get("allowed_base_paths", ["/workspace", "/tmp"])
        self._blocked_paths = self._fileio_config.get(
            "blocked_paths", ["/etc", "/root", "/home", "/usr", "/bin", "/sbin", "/boot", "/dev", "/proc", "/sys"]
        )

        # File operation limits
        self._max_file_size = self._fileio_config.get("max_file_size", 100 * 1024 * 1024)  # 100MB
        self._max_files_per_operation = self._fileio_config.get("max_files_per_operation", 1000)
        self._allowed_extensions = self._fileio_config.get("allowed_extensions", [])  # Empty means all allowed

        # Dangerous file patterns
        self._dangerous_patterns = self._fileio_config.get(
            "dangerous_patterns",
            [
                r"\.\.\/",  # Directory traversal
                r"\/etc\/passwd",
                r"\/etc\/shadow",
                r"\/proc\/",
                r"\/sys\/",
                r"\/dev\/",
            ],
        )

        # Add parameters
        self._setup_parameters()

    def _setup_parameters(self) -> None:
        """Set up file I/O executor parameters."""
        # Operation type parameter
        self.add_parameter(
            Parameter(
                name="operation",
                type=ParameterType.STRING,
                description="Type of file operation to perform",
                required=True,
                enum_values=[
                    "read",
                    "write",
                    "append",
                    "list",
                    "create_dir",
                    "remove",
                    "copy",
                    "move",
                    "exists",
                    "stat",
                    "permissions",
                    "search",
                    "archive",
                    "extract",
                ],
            )
        )

        # File/directory path parameter
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="Target file or directory path",
                required=True,
            )
        )

        # Content parameter for write operations
        self.add_parameter(
            Parameter(
                name="content",
                type=ParameterType.STRING,
                description="Content to write to file (for write/append operations)",
                required=False,
            )
        )

        # Source path for copy/move operations
        self.add_parameter(
            Parameter(
                name="source_path",
                type=ParameterType.STRING,
                description="Source path for copy/move operations",
                required=False,
            )
        )

        # Encoding parameter
        self.add_parameter(
            Parameter(
                name="encoding",
                type=ParameterType.STRING,
                description="Text encoding for file operations",
                required=False,
                default="utf-8",
                enum_values=["utf-8", "ascii", "latin-1", "utf-16"],
            )
        )

        # Recursive parameter for directory operations
        self.add_parameter(
            Parameter(
                name="recursive",
                type=ParameterType.BOOLEAN,
                description="Whether to perform operation recursively",
                required=False,
                default=False,
            )
        )

        # Force parameter for potentially destructive operations
        self.add_parameter(
            Parameter(
                name="force",
                type=ParameterType.BOOLEAN,
                description="Force operation (bypass some safety checks)",
                required=False,
                default=False,
            )
        )

        # Permissions parameter
        self.add_parameter(
            Parameter(
                name="permissions",
                type=ParameterType.STRING,
                description="File permissions in octal format (e.g., '755')",
                required=False,
                pattern=r"^[0-7]{3,4}$",
            )
        )

        # Search pattern parameter
        self.add_parameter(
            Parameter(
                name="pattern",
                type=ParameterType.STRING,
                description="Search pattern (for search operations)",
                required=False,
            )
        )

        # Output format parameter
        self.add_parameter(
            Parameter(
                name="output_format",
                type=ParameterType.STRING,
                description="Format for operation results",
                required=False,
                default="json",
                enum_values=["json", "text", "raw"],
            )
        )

        # Create parent directories parameter
        self.add_parameter(
            Parameter(
                name="create_parents",
                type=ParameterType.BOOLEAN,
                description="Create parent directories if they don't exist",
                required=False,
                default=False,
            )
        )

    def validate_path(self, path: str, operation: str) -> ValidationResult:
        """
        Validate file path for security and accessibility.

        Args:
            path: File/directory path to validate
            operation: Type of operation being performed

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult.success()

        if not path or not path.strip():
            result.add_error("Path cannot be empty")
            return result

        # Normalize path
        normalized_path = os.path.normpath(path)

        # Check for dangerous patterns
        for pattern in self._dangerous_patterns:
            if re.search(pattern, normalized_path, re.IGNORECASE):
                result.add_error(f"Path contains dangerous pattern: {pattern}")

        # Check if path is within allowed base paths
        if self._allowed_base_paths:
            allowed = False
            for base_path in self._allowed_base_paths:
                if normalized_path.startswith(base_path) or normalized_path == base_path:
                    allowed = True
                    break
            if not allowed:
                result.add_error(f"Path must be within allowed base paths: {self._allowed_base_paths}")

        # Check for blocked paths
        for blocked_path in self._blocked_paths:
            if normalized_path.startswith(blocked_path):
                result.add_error(f"Path is in blocked directory: {blocked_path}")

        # Check file extension for write operations
        if operation in ["write", "append"] and self._allowed_extensions:
            _, ext = os.path.splitext(normalized_path)
            if ext and ext.lower() not in self._allowed_extensions:
                result.add_warning(f"File extension '{ext}' may not be allowed")

        return result

    def build_file_command(self, parameters: Dict[str, Any]) -> List[str]:
        """
        Build shell command for file operation.

        Args:
            parameters: Operation parameters

        Returns:
            List of command arguments
        """
        operation = parameters["operation"]
        path = parameters["path"]

        if operation == "read":
            return ["cat", path]

        elif operation == "write":
            content = parameters.get("content", "")
            if parameters.get("create_parents", False):
                return ["sh", "-c", f"mkdir -p $(dirname '{path}') && echo '{content}' > '{path}'"]
            else:
                return ["sh", "-c", f"echo '{content}' > '{path}'"]

        elif operation == "append":
            content = parameters.get("content", "")
            return ["sh", "-c", f"echo '{content}' >> '{path}'"]

        elif operation == "list":
            if parameters.get("recursive", False):
                return ["find", path, "-type", "f"]
            else:
                return ["ls", "-la", path]

        elif operation == "create_dir":
            if parameters.get("recursive", True):
                cmd = ["mkdir", "-p", path]
            else:
                cmd = ["mkdir", path]

            permissions = parameters.get("permissions")
            if permissions:
                cmd.extend(["-m", permissions])
            return cmd

        elif operation == "remove":
            if parameters.get("recursive", False) and parameters.get("force", False):
                return ["rm", "-rf", path]
            elif parameters.get("recursive", False):
                return ["rm", "-r", path]
            elif parameters.get("force", False):
                return ["rm", "-f", path]
            else:
                return ["rm", path]

        elif operation == "copy":
            source = parameters.get("source_path")
            if not source:
                raise SandboxExecutionError("source_path required for copy operation")
            if parameters.get("recursive", False):
                return ["cp", "-r", source, path]
            else:
                return ["cp", source, path]

        elif operation == "move":
            source = parameters.get("source_path")
            if not source:
                raise SandboxExecutionError("source_path required for move operation")
            return ["mv", source, path]

        elif operation == "exists":
            return ["test", "-e", path]

        elif operation == "stat":
            return ["stat", path]

        elif operation == "permissions":
            permissions = parameters.get("permissions")
            if permissions:
                return ["chmod", permissions, path]
            else:
                return ["ls", "-l", path]

        elif operation == "search":
            pattern = parameters.get("pattern", "*")
            return ["find", path, "-name", pattern]

        elif operation == "archive":
            # Create tar archive
            return ["tar", "-czf", f"{path}.tar.gz", "-C", os.path.dirname(path), os.path.basename(path)]

        elif operation == "extract":
            # Extract archive based on extension
            if path.endswith(".tar.gz") or path.endswith(".tgz"):
                return ["tar", "-xzf", path]
            elif path.endswith(".zip"):
                return ["unzip", path]
            else:
                raise SandboxExecutionError(f"Unsupported archive format: {path}")

        else:
            raise SandboxExecutionError(f"Unsupported operation: {operation}")

    def parse_file_output(
        self, stdout: str, stderr: str, return_code: int, operation: str, path: str, output_format: str
    ) -> CommandResult:
        """
        Parse file operation output into a structured result.

        Args:
            stdout: Standard output from command
            stderr: Standard error from command
            return_code: Process exit code
            operation: File operation performed
            path: Target path
            output_format: Requested output format

        Returns:
            CommandResult with structured file operation results
        """
        success = return_code == 0

        result_data = {
            "operation": operation,
            "path": path,
            "success": success,
            "return_code": return_code,
            "output_format": output_format,
        }

        if success:
            if operation == "read":
                if output_format == "json":
                    result_data["content"] = stdout
                    result_data["size"] = len(stdout)
                else:
                    result_data["content"] = stdout

            elif operation in ["write", "append"]:
                result_data["message"] = "File operation completed successfully"
                if stdout.strip():
                    result_data["output"] = stdout.strip()

            elif operation == "list":
                if output_format == "json":
                    lines = [line.strip() for line in stdout.split("\n") if line.strip()]
                    if operation == "list" and not any("total" in line for line in lines[:2]):
                        # Simple file list
                        result_data["files"] = lines
                    else:
                        # Detailed ls output
                        files = []
                        for line in lines:
                            if line.startswith("total"):
                                continue
                            parts = line.split()
                            if len(parts) >= 9:
                                files.append(
                                    {
                                        "permissions": parts[0],
                                        "links": parts[1],
                                        "owner": parts[2],
                                        "group": parts[3],
                                        "size": parts[4],
                                        "date": " ".join(parts[5:8]),
                                        "name": " ".join(parts[8:]),
                                    }
                                )
                        result_data["files"] = files
                else:
                    result_data["content"] = stdout

            elif operation == "stat":
                if output_format == "json":
                    # Parse stat output
                    stat_info = {}
                    for line in stdout.split("\n"):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            stat_info[key.strip()] = value.strip()
                    result_data["stat_info"] = stat_info
                else:
                    result_data["content"] = stdout

            elif operation == "exists":
                result_data["exists"] = True

            elif operation == "search":
                if output_format == "json":
                    files = []
                    for line in stdout.split("\n"):
                        if line.strip():
                            # Create dictionary with file path information
                            files.append({"path": line.strip(), "name": os.path.basename(line.strip())})
                    result_data["matches"] = files
                    result_data["count"] = len(files)
                else:
                    result_data["content"] = stdout

            else:
                result_data["message"] = f"{operation} operation completed successfully"
                if stdout.strip():
                    result_data["output"] = stdout.strip()

        else:
            # Handle specific error cases
            if operation == "exists" and return_code == 1:
                result_data["exists"] = False
                success = True  # This is expected behavior for 'test -e'
                result_data["success"] = True

        # Add raw output for debugging
        if stderr.strip():
            result_data["stderr"] = stderr.strip()
        if stdout.strip() and operation not in ["read", "list", "stat", "search"]:
            result_data["stdout"] = stdout.strip()

        metadata = {
            "command_type": "file_io",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            error_msg = f"File operation '{operation}' failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute file I/O operation in Docker container.

        Args:
            parameters: Execution parameters including operation and path
            context: Execution context including session_id

        Returns:
            CommandResult with file operation results
        """
        try:
            # Extract session ID
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for file I/O execution")

            # Get Docker environment
            environment = self.get_session_environment(session_id)

            # Validate path
            operation = parameters["operation"]
            path_validation = self.validate_path(parameters["path"], operation)
            if not path_validation.valid:
                return CommandResult.error_result(error=f"Path validation failed: {', '.join(path_validation.errors)}")

            # Log any warnings
            if path_validation.warnings:
                logger.warning(f"Path warnings: {', '.join(path_validation.warnings)}")

            # Validate source path for copy/move operations
            if operation in ["copy", "move"]:
                source_path = parameters.get("source_path")
                if source_path:
                    source_validation = self.validate_path(source_path, operation)
                    if not source_validation.valid:
                        return CommandResult.error_result(
                            error=f"Source path validation failed: {', '.join(source_validation.errors)}"
                        )

            # Check file size for write operations
            if operation in ["write", "append"]:
                content = parameters.get("content", "")
                if len(content.encode("utf-8")) > self._max_file_size:
                    return CommandResult.error_result(
                        error=f"Content size exceeds maximum allowed size of {self._max_file_size} bytes"
                    )

            # Build file command
            try:
                file_cmd = self.build_file_command(parameters)
            except Exception as e:
                return CommandResult.error_result(error=f"Failed to build file command: {e}")

            # Execute file command
            result = environment.execute_command(command=file_cmd)

            # Parse output
            output_format = parameters.get("output_format", "json")
            tool_result = self.parse_file_output(
                result.stdout, result.stderr, result.exit_code, operation, parameters["path"], output_format
            )

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "session_id": session_id,
                    "execution_time": result.execution_time,
                    "file_operation": operation,
                    "target_path": parameters["path"],
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"File I/O execution error: {e}")
            return CommandResult.error_result(f"File I/O execution failed: {str(e)}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for file I/O execution.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with comprehensive validation
        """
        # Run base Docker validation
        result = super().validate_parameters(parameters)

        # Add file I/O-specific validation
        operation = parameters.get("operation")
        if operation:
            # Validate path
            path_validation = self.validate_path(parameters.get("path", ""), operation)
            result.errors.extend(path_validation.errors)
            result.warnings.extend(path_validation.warnings)

            # Validate operation-specific requirements
            if operation in ["copy", "move"] and "source_path" not in parameters:
                result.add_error(f"{operation} operation requires source_path parameter")

            if operation in ["write", "append"] and "content" not in parameters:
                result.add_error(f"{operation} operation requires content parameter")

            if operation == "search" and "pattern" not in parameters:
                result.add_error("search operation requires pattern parameter")

        # Validate permissions format
        permissions = parameters.get("permissions")
        if permissions and not re.match(r"^[0-7]{3,4}$", str(permissions)):
            result.add_error("permissions must be in octal format (e.g., '755')")

        return result
