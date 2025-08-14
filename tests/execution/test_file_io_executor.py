"""
Tests for File I/O executor.

This module tests the secure file I/O executor that handles file and directory operations
in Docker containers with proper security validation and path restrictions.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from saber.server.execution.base import CommandResult, ParameterType, ValidationResult
from saber.server.execution.executors.file_io_executor import FileIoExecutor
from saber.server.execution.sandbox.sandbox_manager import SandboxManager
from saber.server.execution.exceptions import SandboxExecutionError


class TestFileIoExecutor:
    """Test cases for File I/O executor."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxManager."""
        manager = MagicMock(spec=SandboxManager)
        manager.get_sandbox_config.return_value = {
            "image": "saber/fileio-sandbox:latest",
            "network_mode": "none",
            "read_only_root": True,
            "user": "tooluser:tooluser"
        }
        return manager

    @pytest.fixture
    def fileio_executor(self, mock_sandbox_manager):
        """Create a File I/O executor instance for testing."""
        return FileIoExecutor(sandbox_manager=mock_sandbox_manager, timeout=60.0)

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
        """Test file I/O executor initialization."""
        executor = FileIoExecutor(sandbox_manager=mock_sandbox_manager)

        assert executor._sandbox_manager == mock_sandbox_manager
        assert "/workspace" in executor._allowed_base_paths
        assert "/tmp" in executor._allowed_base_paths
        assert "/etc" in executor._blocked_paths
        assert "/root" in executor._blocked_paths
        assert executor._max_file_size == 100 * 1024 * 1024

    def test_initialization_with_config(self, mock_sandbox_manager):
        """Test file I/O executor initialization with custom config."""
        config = {
            "allowed_base_paths": ["/data"],
            "blocked_paths": ["/system"],
            "max_file_size": 50 * 1024 * 1024,
            "allowed_extensions": [".txt", ".json"]
        }
        executor = FileIoExecutor(sandbox_manager=mock_sandbox_manager, fileio_config=config)

        assert executor._allowed_base_paths == ["/data"]
        assert executor._blocked_paths == ["/system"]
        assert executor._max_file_size == 50 * 1024 * 1024
        assert executor._allowed_extensions == [".txt", ".json"]

    def test_parameters_setup(self, fileio_executor):
        """Test that file I/O executor sets up required parameters."""
        params = fileio_executor.get_parameters()

        assert "operation" in params
        assert params["operation"].required is True
        assert "read" in params["operation"].enum_values
        assert "write" in params["operation"].enum_values

        assert "path" in params
        assert params["path"].required is True
        assert params["path"].type == ParameterType.STRING

        assert "content" in params
        assert params["content"].required is False

        assert "recursive" in params
        assert params["recursive"].default is False

    def test_validate_path_valid(self, fileio_executor):
        """Test path validation with valid paths."""
        valid_paths = [
            "/workspace/data.txt",
            "/tmp/temp_file.log",
            "/workspace/subdir/file.json",
        ]

        for path in valid_paths:
            result = fileio_executor.validate_path(path, "read")
            assert result.valid, f"Path should be valid: {path}"

    def test_validate_path_blocked(self, fileio_executor):
        """Test path validation with blocked paths."""
        blocked_paths = [
            "/etc/passwd",
            "/root/.ssh/id_rsa",
            "/usr/bin/sudo",
            "/proc/cpuinfo",
        ]

        for path in blocked_paths:
            result = fileio_executor.validate_path(path, "read")
            assert not result.valid, f"Path should be blocked: {path}"

    def test_validate_path_directory_traversal(self, fileio_executor):
        """Test path validation with directory traversal attempts."""
        malicious_paths = [
            "/workspace/../etc/passwd",
            "/tmp/../../root/.ssh/id_rsa",
            "../../../etc/shadow",
        ]

        for path in malicious_paths:
            result = fileio_executor.validate_path(path, "read")
            assert not result.valid, f"Path should be blocked (traversal): {path}"

    def test_validate_path_empty(self, fileio_executor):
        """Test path validation with empty path."""
        result = fileio_executor.validate_path("", "read")
        assert not result.valid
        assert "cannot be empty" in result.errors[0]

    def test_validate_path_outside_allowed_base(self, fileio_executor):
        """Test path validation outside allowed base paths."""
        result = fileio_executor.validate_path("/home/user/file.txt", "read")
        assert not result.valid
        assert "must be within allowed base paths" in result.errors[0]

    def test_build_file_command_read(self, fileio_executor):
        """Test building command for file read operation."""
        parameters = {
            "operation": "read",
            "path": "/workspace/data.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["cat", "/workspace/data.txt"]

    def test_build_file_command_write(self, fileio_executor):
        """Test building command for file write operation."""
        parameters = {
            "operation": "write",
            "path": "/workspace/output.txt",
            "content": "Hello, World!"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert "sh" in cmd
        assert "-c" in cmd
        assert "echo 'Hello, World!' > '/workspace/output.txt'" in " ".join(cmd)

    def test_build_file_command_write_with_parents(self, fileio_executor):
        """Test building command for file write with parent directory creation."""
        parameters = {
            "operation": "write",
            "path": "/workspace/new/dir/file.txt",
            "content": "test",
            "create_parents": True
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert "mkdir -p" in " ".join(cmd)

    def test_build_file_command_append(self, fileio_executor):
        """Test building command for file append operation."""
        parameters = {
            "operation": "append",
            "path": "/workspace/log.txt",
            "content": "New log entry"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert ">>" in " ".join(cmd)

    def test_build_file_command_list(self, fileio_executor):
        """Test building command for directory listing."""
        parameters = {
            "operation": "list",
            "path": "/workspace"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["ls", "-la", "/workspace"]

    def test_build_file_command_list_recursive(self, fileio_executor):
        """Test building command for recursive directory listing."""
        parameters = {
            "operation": "list",
            "path": "/workspace",
            "recursive": True
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["find", "/workspace", "-type", "f"]

    def test_build_file_command_create_dir(self, fileio_executor):
        """Test building command for directory creation."""
        parameters = {
            "operation": "create_dir",
            "path": "/workspace/newdir",
            "permissions": "755"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert "mkdir" in cmd
        assert "-p" in cmd
        assert "/workspace/newdir" in cmd
        assert "-m" in cmd
        assert "755" in cmd

    def test_build_file_command_remove(self, fileio_executor):
        """Test building command for file removal."""
        parameters = {
            "operation": "remove",
            "path": "/workspace/temp.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["rm", "/workspace/temp.txt"]

    def test_build_file_command_remove_recursive_force(self, fileio_executor):
        """Test building command for recursive forced removal."""
        parameters = {
            "operation": "remove",
            "path": "/workspace/tempdir",
            "recursive": True,
            "force": True
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["rm", "-rf", "/workspace/tempdir"]

    def test_build_file_command_copy(self, fileio_executor):
        """Test building command for file copy."""
        parameters = {
            "operation": "copy",
            "source_path": "/workspace/source.txt",
            "path": "/workspace/dest.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["cp", "/workspace/source.txt", "/workspace/dest.txt"]

    def test_build_file_command_move(self, fileio_executor):
        """Test building command for file move."""
        parameters = {
            "operation": "move",
            "source_path": "/workspace/old.txt",
            "path": "/workspace/new.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["mv", "/workspace/old.txt", "/workspace/new.txt"]

    def test_build_file_command_exists(self, fileio_executor):
        """Test building command for file existence check."""
        parameters = {
            "operation": "exists",
            "path": "/workspace/file.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["test", "-e", "/workspace/file.txt"]

    def test_build_file_command_stat(self, fileio_executor):
        """Test building command for file stat."""
        parameters = {
            "operation": "stat",
            "path": "/workspace/file.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["stat", "/workspace/file.txt"]

    def test_build_file_command_search(self, fileio_executor):
        """Test building command for file search."""
        parameters = {
            "operation": "search",
            "path": "/workspace",
            "pattern": "*.txt"
        }

        cmd = fileio_executor.build_file_command(parameters)

        assert cmd == ["find", "/workspace", "-name", "*.txt"]

    def test_build_file_command_unsupported(self, fileio_executor):
        """Test building command for unsupported operation."""
        parameters = {
            "operation": "unsupported",
            "path": "/workspace/file.txt"
        }

        with pytest.raises(SandboxExecutionError) as excinfo:
            fileio_executor.build_file_command(parameters)

        assert "Unsupported operation" in str(excinfo.value)

    def test_parse_file_output_read_success(self, fileio_executor):
        """Test parsing file read output."""
        stdout = "Hello, World!\nThis is a test file."
        stderr = ""
        return_code = 0

        result = fileio_executor.parse_file_output(stdout, stderr, return_code, "read", "/workspace/file.txt", "json")

        assert result.success
        assert result.data["content"] == stdout
        assert result.data["size"] == len(stdout)

    def test_parse_file_output_list_detailed(self, fileio_executor):
        """Test parsing detailed directory listing output."""
        stdout = """total 8
-rw-r--r-- 1 user user 100 Jan 15 10:30 file1.txt
-rw-r--r-- 1 user user 200 Jan 15 10:31 file2.txt"""
        stderr = ""
        return_code = 0

        result = fileio_executor.parse_file_output(stdout, stderr, return_code, "list", "/workspace", "json")

        assert result.success
        assert "files" in result.data
        assert len(result.data["files"]) == 2
        assert result.data["files"][0]["name"] == "file1.txt"
        assert result.data["files"][0]["size"] == "100"

    def test_parse_file_output_exists_true(self, fileio_executor):
        """Test parsing file exists output when file exists."""
        stdout = ""
        stderr = ""
        return_code = 0

        result = fileio_executor.parse_file_output(stdout, stderr, return_code, "exists", "/workspace/file.txt", "json")

        assert result.success
        assert result.data["exists"] is True

    def test_parse_file_output_exists_false(self, fileio_executor):
        """Test parsing file exists output when file doesn't exist."""
        stdout = ""
        stderr = ""
        return_code = 1

        result = fileio_executor.parse_file_output(stdout, stderr, return_code, "exists", "/workspace/file.txt", "json")

        assert result.success  # This is expected behavior
        assert result.data["exists"] is False

    def test_parse_file_output_error(self, fileio_executor):
        """Test parsing file operation output with error."""
        stdout = ""
        stderr = "Permission denied"
        return_code = 1

        result = fileio_executor.parse_file_output(stdout, stderr, return_code, "read", "/etc/passwd", "json")

        assert not result.success
        assert "Permission denied" in result.error

    @pytest.mark.asyncio
    async def test_execute_read_success(self, fileio_executor, mock_docker_environment):
        """Test successful file read execution."""
        parameters = {
            "operation": "read",
            "path": "/workspace/test.txt"
        }
        context = {"session_id": "test-session"}

        fileio_executor.get_session_environment = MagicMock(return_value=mock_docker_environment)

        mock_result = MagicMock()
        mock_result.stdout = "File content here"
        mock_result.stderr = ""
        mock_result.exit_code = 0
        mock_result.execution_time = 0.1
        mock_docker_environment.execute_command.return_value = mock_result

        result = await fileio_executor.execute(parameters, context)

        assert result.success
        assert result.data["content"] == "File content here"
        assert result.metadata["session_id"] == "test-session"

    @pytest.mark.asyncio
    async def test_execute_write_large_content(self, fileio_executor):
        """Test file write with content exceeding size limit."""
        large_content = "x" * (101 * 1024 * 1024)  # 101MB
        parameters = {
            "operation": "write",
            "path": "/workspace/large.txt",
            "content": large_content
        }
        context = {"session_id": "test-session"}

        result = await fileio_executor.execute(parameters, context)

        assert not result.success
        assert "exceeds maximum allowed size" in result.error

    @pytest.mark.asyncio
    async def test_execute_path_validation_failure(self, fileio_executor):
        """Test file operation with path validation failure."""
        parameters = {
            "operation": "read",
            "path": "/etc/passwd"
        }
        context = {"session_id": "test-session"}

        result = await fileio_executor.execute(parameters, context)

        assert not result.success
        assert "Path validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_copy_source_validation(self, fileio_executor):
        """Test copy operation with source path validation."""
        parameters = {
            "operation": "copy",
            "source_path": "/etc/passwd",
            "path": "/workspace/copy.txt"
        }
        context = {"session_id": "test-session"}

        result = await fileio_executor.execute(parameters, context)

        assert not result.success
        assert "Source path validation failed" in result.error

    @pytest.mark.asyncio
    async def test_execute_missing_session_id(self, fileio_executor):
        """Test file operation without session_id."""
        parameters = {
            "operation": "read",
            "path": "/workspace/file.txt"
        }
        context = {}

        result = await fileio_executor.execute(parameters, context)

        assert not result.success
        assert "session_id required" in result.error

    def test_validate_parameters_success(self, fileio_executor):
        """Test parameter validation with valid parameters."""
        parameters = {
            "operation": "write",
            "path": "/workspace/file.txt",
            "content": "Hello, World!",
            "permissions": "644",
            "create_parents": True
        }

        result = fileio_executor.validate_parameters(parameters)
        assert result.valid

    def test_validate_parameters_copy_missing_source(self, fileio_executor):
        """Test parameter validation for copy without source_path."""
        parameters = {
            "operation": "copy",
            "path": "/workspace/dest.txt"
        }

        result = fileio_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("copy operation requires source_path" in error for error in result.errors)

    def test_validate_parameters_write_missing_content(self, fileio_executor):
        """Test parameter validation for write without content."""
        parameters = {
            "operation": "write",
            "path": "/workspace/file.txt"
        }

        result = fileio_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("write operation requires content" in error for error in result.errors)

    def test_validate_parameters_search_missing_pattern(self, fileio_executor):
        """Test parameter validation for search without pattern."""
        parameters = {
            "operation": "search",
            "path": "/workspace"
        }

        result = fileio_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("search operation requires pattern" in error for error in result.errors)

    def test_validate_parameters_invalid_permissions(self, fileio_executor):
        """Test parameter validation with invalid permissions."""
        parameters = {
            "operation": "create_dir",
            "path": "/workspace/dir",
            "permissions": "999"  # Invalid octal
        }

        result = fileio_executor.validate_parameters(parameters)
        assert not result.valid
        assert any("permissions must be in octal format" in error for error in result.errors)

    def test_to_mcp_schema(self, fileio_executor):
        """Test MCP schema generation."""
        schema = fileio_executor.to_mcp_schema()

        assert schema["type"] == "object"
        assert "properties" in schema
        assert "operation" in schema["properties"]
        assert "path" in schema["properties"]
        assert "required" in schema
        assert "operation" in schema["required"]
        assert "path" in schema["required"]

    def test_security_command_metadata(self, fileio_executor):
        """Test security command metadata."""
        metadata = fileio_executor._security_command_metadata

        assert metadata["domain"] == "filesystem"
        assert metadata["name"] == "file_io"
        assert metadata["security_level"] == "medium"
        assert metadata["requires_validation"] is True
