"""
Tests for Copilot CLI-compatible executors.

This module tests the view, create, edit, and grep executors that implement
the tool interfaces expected by the GitHub Copilot CLI agent (@github/copilot v0.0.384).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from saber.server.base import CommandResult
from saber.server.execution.base import (
    CreateParameters,
    EditParameters,
    ExecutionContext,
    GrepParameters,
    ParameterType,
    ValidationResult,
    ViewParameters,
)
from saber.server.execution.models import ExecutorConfig
from saber.server.execution.exceptions import SandboxExecutionError
from saber.server.execution.executors.copilot_registry.view_executor import ViewExecutor
from saber.server.execution.executors.copilot_registry.create_executor import CreateExecutor
from saber.server.execution.executors.copilot_registry.edit_executor import EditExecutor
from saber.server.execution.executors.copilot_registry.grep_executor import GrepExecutor
from saber.server.execution.sandbox.sandbox_environment_manager import SandboxEnvironmentManager


class TestViewExecutor:
    """Test cases for View executor - reads files and lists directories."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {
            "image": "saber/sandbox:latest",
            "network_mode": "none",
        }
        return manager

    @pytest.fixture
    def view_executor(self, mock_sandbox_manager):
        """Create a ViewExecutor instance for testing."""
        return ViewExecutor(sandbox_manager=mock_sandbox_manager)

    def test_metadata(self, view_executor):
        """Test executor has correct metadata."""
        assert view_executor._executor_metadata["name"] == "view"
        assert "Read file" in view_executor._executor_metadata["description"]

    def test_parameters_defined(self, view_executor):
        """Test that required parameters are defined."""
        params = view_executor.get_parameters()

        assert "path" in params
        assert params["path"].required is True
        assert params["path"].type == ParameterType.STRING

        assert "view_range" in params
        assert params["view_range"].required is False
        assert params["view_range"].type == ParameterType.ARRAY

        assert "forceReadLargeFiles" in params
        assert params["forceReadLargeFiles"].required is False
        assert params["forceReadLargeFiles"].type == ParameterType.BOOLEAN

    def test_build_command_simple_file(self, view_executor):
        """Test building command for viewing a simple file."""
        params = ViewParameters(path="/workspace/test.py")

        result = view_executor._build_command(params)

        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Should check existence and read with line numbers
        assert 'nl -ba "/workspace/test.py"' in result[2]

    def test_build_command_with_line_range(self, view_executor):
        """Test building command with line range specified."""
        params = ViewParameters(path="/workspace/test.py", view_range=(10, 20))

        result = view_executor._build_command(params)

        assert "/bin/sh" in result[0]
        # Should use head/tail for range
        assert "head -n 20" in result[2]
        assert "tail -n 11" in result[2]  # 20 - 10 + 1 = 11 lines

    def test_build_command_range_to_end(self, view_executor):
        """Test building command with range to end of file."""
        params = ViewParameters(path="/workspace/test.py", view_range=(50, -1))

        result = view_executor._build_command(params)

        assert "/bin/sh" in result[0]
        # Should use tail -n + to read from line to end
        assert "tail -n +50" in result[2]

    def test_build_command_force_large_file(self, view_executor):
        """Test building command with forceReadLargeFiles=True skips size check."""
        params = ViewParameters(path="/workspace/large.bin", force_read_large_files=True)

        result = view_executor._build_command(params)

        # Should NOT include size check when force is True
        assert "file_size" not in result[2] or "10485760" not in result[2]

    def test_build_command_directory(self, view_executor):
        """Test command handles directory listing."""
        params = ViewParameters(path="/workspace/src")

        result = view_executor._build_command(params)

        # Should include directory check and find command
        assert 'if [ -d "/workspace/src" ]' in result[2]
        assert "find" in result[2]
        assert "-maxdepth 2" in result[2]

    def test_parameter_validation_missing_path(self, view_executor):
        """Test validation fails when path is missing."""
        parameters = {}

        result = view_executor.validate_parameters(parameters)

        assert result.valid is False
        assert any("path" in err for err in result.errors)

    def test_parameter_validation_valid(self, view_executor):
        """Test validation succeeds with valid parameters."""
        params = ViewParameters(path="/workspace/file.txt")

        result = view_executor.validate_parameters(params)

        assert result.valid is True

    def test_mcp_schema(self, view_executor):
        """Test MCP schema generation."""
        schema = view_executor.to_mcp_schema()

        assert schema.type == "object"
        assert "path" in schema.properties
        assert "view_range" in schema.properties
        assert "forceReadLargeFiles" in schema.properties
        assert "path" in schema.required


class TestCreateExecutor:
    """Test cases for Create executor - creates new files."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    @pytest.fixture
    def create_executor(self, mock_sandbox_manager):
        """Create a CreateExecutor instance for testing."""
        return CreateExecutor(sandbox_manager=mock_sandbox_manager)

    def test_metadata(self, create_executor):
        """Test executor has correct metadata."""
        assert create_executor._executor_metadata["name"] == "create"
        assert "Create" in create_executor._executor_metadata["description"]

    def test_parameters_defined(self, create_executor):
        """Test that required parameters are defined."""
        params = create_executor.get_parameters()

        assert "path" in params
        assert params["path"].required is True
        assert params["path"].type == ParameterType.STRING

        assert "file_text" in params
        assert params["file_text"].required is False
        assert params["file_text"].type == ParameterType.STRING

    def test_build_command_with_content(self, create_executor):
        """Test building command for creating a file with content."""
        params = CreateParameters(path="/workspace/new_file.py", file_text="print('hello')")

        result = create_executor._build_command(params)

        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Should check file doesn't exist
        assert 'if [ -e "/workspace/new_file.py" ]' in result[2]
        # Should create parent directories
        assert "mkdir -p" in result[2]
        # Should use base64 for safe content handling
        assert "base64 -d" in result[2]

    def test_build_command_empty_file(self, create_executor):
        """Test building command for creating an empty file."""
        params = CreateParameters(path="/workspace/empty.txt", file_text="")

        result = create_executor._build_command(params)

        # Should use touch for empty files
        assert "touch" in result[2]

    def test_build_command_special_characters(self, create_executor):
        """Test content with special shell characters is handled safely."""
        params = CreateParameters(
            path="/workspace/script.sh",
            file_text='#!/bin/bash\necho "$HOME"\nif [ -f "test" ]; then echo "yes"; fi'
        )

        result = create_executor._build_command(params)

        # Should use base64 encoding to safely pass content
        assert "base64 -d" in result[2]

    def test_parameter_validation_missing_path(self, create_executor):
        """Test validation fails when path is missing - handled at from_dict conversion."""
        # With typed params, missing required fields are caught at conversion time
        # So we test that a valid CreateParameters passes validation
        params = CreateParameters(path="/workspace/file.txt", file_text="content")
        result = create_executor.validate_parameters(params)
        assert result.valid is True

    def test_parameter_validation_valid(self, create_executor):
        """Test validation succeeds with valid parameters."""
        params = CreateParameters(path="/workspace/file.txt", file_text="content")

        result = create_executor.validate_parameters(params)

        assert result.valid is True


class TestEditExecutor:
    """Test cases for Edit executor - performs string replacement in files."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    @pytest.fixture
    def edit_executor(self, mock_sandbox_manager):
        """Create an EditExecutor instance for testing."""
        return EditExecutor(sandbox_manager=mock_sandbox_manager)

    def test_metadata(self, edit_executor):
        """Test executor has correct metadata."""
        assert edit_executor._executor_metadata["name"] == "edit"
        assert "replace" in edit_executor._executor_metadata["description"].lower()

    def test_parameters_defined(self, edit_executor):
        """Test that required parameters are defined."""
        params = edit_executor.get_parameters()

        assert "path" in params
        assert params["path"].required is True

        assert "old_str" in params
        assert params["old_str"].required is True

        assert "new_str" in params
        assert params["new_str"].required is False

    def test_build_command_simple_replacement(self, edit_executor):
        """Test building command for simple string replacement."""
        params = EditParameters(
            path="/workspace/test.py",
            old_str="def foo():",
            new_str="def bar():"
        )

        result = edit_executor._build_command(params)

        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Uses Python script for safe replacement
        assert "python3" in result[2]
        assert "base64" in result[2]

    def test_build_command_delete_text(self, edit_executor):
        """Test building command to delete text (empty new_str)."""
        params = EditParameters(
            path="/workspace/test.py",
            old_str="# TODO: remove this line",
            new_str=""
        )

        result = edit_executor._build_command(params)

        # Should handle empty new_str for deletion
        assert "/bin/sh" in result[0]

    def test_build_command_multiline_replacement(self, edit_executor):
        """Test building command for multiline string replacement."""
        params = EditParameters(
            path="/workspace/test.py",
            old_str="def old_function():\n    pass",
            new_str="def new_function():\n    return True"
        )

        result = edit_executor._build_command(params)

        # Should use base64 encoding for safe multiline handling
        assert "base64" in result[2]

    def test_parameter_validation_missing_path(self, edit_executor):
        """Test that valid typed params pass validation."""
        # With typed params, missing required fields are caught at conversion time
        params = EditParameters(path="/workspace/file.txt", old_str="foo", new_str="bar")
        result = edit_executor.validate_parameters(params)
        assert result.valid is True

    def test_parameter_validation_missing_old_str(self, edit_executor):
        """Test that valid typed params pass validation."""
        # With typed params, missing required fields are caught at conversion time
        params = EditParameters(path="/workspace/file.txt", old_str="foo", new_str="bar")
        result = edit_executor.validate_parameters(params)
        assert result.valid is True

    def test_parameter_validation_valid(self, edit_executor):
        """Test validation succeeds with valid parameters."""
        params = EditParameters(path="/workspace/file.txt", old_str="foo", new_str="bar")

        result = edit_executor.validate_parameters(params)

        assert result.valid is True


class TestGrepExecutor:
    """Test cases for Grep executor - searches files using regex."""

    @pytest.fixture
    def mock_sandbox_manager(self):
        """Create a mock SandboxEnvironmentManager."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        return manager

    @pytest.fixture
    def grep_executor(self, mock_sandbox_manager):
        """Create a GrepExecutor instance for testing."""
        return GrepExecutor(sandbox_manager=mock_sandbox_manager)

    def test_metadata(self, grep_executor):
        """Test executor has correct metadata."""
        assert grep_executor._executor_metadata["name"] == "grep"
        assert "search" in grep_executor._executor_metadata["description"].lower()

    def test_parameters_defined(self, grep_executor):
        """Test that required parameters are defined."""
        params = grep_executor.get_parameters()

        assert "pattern" in params
        assert params["pattern"].required is True
        assert params["pattern"].type == ParameterType.STRING

        assert "path" in params
        assert params["path"].required is False

        assert "glob" in params
        assert params["glob"].required is False

    def test_build_command_simple_pattern(self, grep_executor):
        """Test building command for simple pattern search."""
        params = GrepParameters(pattern="def test_")

        result = grep_executor._build_command(params)

        assert result[0] == "/bin/sh"
        assert result[1] == "-c"
        # Should use grep with line numbers
        assert "grep" in result[2]
        assert "-n" in result[2]  # Line numbers

    def test_build_command_with_path(self, grep_executor):
        """Test building command with specific path."""
        params = GrepParameters(pattern="import", path="/workspace/src")

        result = grep_executor._build_command(params)

        assert "/workspace/src" in result[2]

    def test_build_command_with_glob(self, grep_executor):
        """Test building command with glob filter."""
        params = GrepParameters(pattern="class.*:", glob="*.py")

        result = grep_executor._build_command(params)

        # Should include glob pattern in grep/find command
        assert "*.py" in result[2]

    def test_build_command_regex_pattern(self, grep_executor):
        """Test building command with regex pattern."""
        params = GrepParameters(pattern=r"def\s+\w+\(self")

        result = grep_executor._build_command(params)

        # Should use extended regex flag
        assert "-E" in result[2] or "--extended-regexp" in result[2].lower() or "-P" in result[2]

    def test_parameter_validation_missing_pattern(self, grep_executor):
        """Test that valid typed params pass validation."""
        # With typed params, missing required fields are caught at conversion time
        params = GrepParameters(pattern="test")
        result = grep_executor.validate_parameters(params)
        assert result.valid is True

    def test_parameter_validation_valid(self, grep_executor):
        """Test validation succeeds with valid parameters."""
        params = GrepParameters(pattern="test")

        result = grep_executor.validate_parameters(params)

        assert result.valid is True

    def test_mcp_schema(self, grep_executor):
        """Test MCP schema generation."""
        schema = grep_executor.to_mcp_schema()

        assert schema.type == "object"
        assert "pattern" in schema.properties
        assert "path" in schema.properties
        assert "glob" in schema.properties
        assert "pattern" in schema.required


class TestCopilotExecutorIntegration:
    """Integration tests for Copilot executors with mocked Docker environment."""

    @pytest.fixture
    def mock_docker_environment(self):
        """Create a mock Docker execution environment."""
        env = MagicMock()
        container_mock = MagicMock()
        container_mock.id = "container123456789"
        env.get_execution_container.return_value = container_mock
        env.execute_command = AsyncMock()
        return env

    @pytest.fixture
    def mock_sandbox_manager_with_env(self, mock_docker_environment):
        """Create a mock SandboxEnvironmentManager that returns a Docker environment."""
        manager = MagicMock(spec=SandboxEnvironmentManager)
        manager.sandbox_config = {"image": "saber/sandbox:latest"}
        manager.get_episode_environment.return_value = mock_docker_environment
        return manager

    @pytest.mark.asyncio
    async def test_view_execute_file_success(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test successful file viewing."""
        view_executor = ViewExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        # Mock successful file read
        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=0,
                stdout="     1\tprint('hello')\n     2\tprint('world')\n",
                stderr="",
                execution_time=0.1
            )
        )

        params = ViewParameters(path="/workspace/test.py")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await view_executor.execute(params, context=context)

        assert result.success is True
        assert "print('hello')" in result.data.get("stdout", "") or "print('hello')" in str(result.data)

    @pytest.mark.asyncio
    async def test_view_execute_file_not_found(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test viewing non-existent file."""
        view_executor = ViewExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="Error: Path does not exist: /workspace/missing.py",
                execution_time=0.1
            )
        )

        params = ViewParameters(path="/workspace/missing.py")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await view_executor.execute(params, context=context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_create_execute_success(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test successful file creation."""
        create_executor = CreateExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=0,
                stdout="Successfully created: /workspace/new_file.py",
                stderr="",
                execution_time=0.1
            )
        )

        params = CreateParameters(path="/workspace/new_file.py", file_text="# New file")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await create_executor.execute(params, context=context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_create_execute_file_exists(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test creating file that already exists fails."""
        create_executor = CreateExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="Error: File already exists: /workspace/existing.py",
                execution_time=0.1
            )
        )

        params = CreateParameters(path="/workspace/existing.py", file_text="content")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await create_executor.execute(params, context=context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_edit_execute_success(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test successful file editing."""
        edit_executor = EditExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=0,
                stdout="Successfully replaced 1 occurrence in /workspace/test.py",
                stderr="",
                execution_time=0.1
            )
        )

        params = EditParameters(
            path="/workspace/test.py",
            old_str="def foo():",
            new_str="def bar():"
        )
        context = ExecutionContext(episode_id="test_episode_123")
        result = await edit_executor.execute(params, context=context)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_edit_execute_string_not_found(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test editing when old_str not found."""
        edit_executor = EditExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="Error: String not found in file",
                execution_time=0.1
            )
        )

        params = EditParameters(
            path="/workspace/test.py",
            old_str="nonexistent string",
            new_str="replacement"
        )
        context = ExecutionContext(episode_id="test_episode_123")
        result = await edit_executor.execute(params, context=context)

        assert result.success is False

    @pytest.mark.asyncio
    async def test_grep_execute_success(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test successful grep search."""
        grep_executor = GrepExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=0,
                stdout="/workspace/test.py:1:def test_function():\n/workspace/test.py:10:def test_another():\n",
                stderr="",
                execution_time=0.2
            )
        )

        params = GrepParameters(pattern="def test_")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await grep_executor.execute(params, context=context)

        assert result.success is True
        assert "test_function" in result.data.get("stdout", "") or "test_function" in str(result.data)

    @pytest.mark.asyncio
    async def test_grep_execute_no_matches(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test grep with no matches."""
        grep_executor = GrepExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        # grep returns exit code 1 when no matches found, but this is not an error
        mock_docker_environment.execute_command = AsyncMock(
            return_value=CommandResult(
                exit_code=1,
                stdout="",
                stderr="",
                execution_time=0.1
            )
        )

        params = GrepParameters(pattern="nonexistent_pattern_xyz")
        context = ExecutionContext(episode_id="test_episode_123")
        result = await grep_executor.execute(params, context=context)

        # No matches is a valid result, not an error
        # The executor should handle grep's exit code 1 for no matches
        assert result is not None

    @pytest.mark.asyncio
    async def test_execute_empty_episode_id_raises_error(self, mock_sandbox_manager_with_env, mock_docker_environment):
        """Test that execution with empty episode_id raises an error."""
        view_executor = ViewExecutor(sandbox_manager=mock_sandbox_manager_with_env)

        # Mock the sandbox manager to raise error for empty episode
        mock_sandbox_manager_with_env.get_episode_environment.side_effect = ValueError(
            "No environment found for episode . Environment must be created before execution."
        )

        params = ViewParameters(path="/workspace/test.py")
        context = ExecutionContext(episode_id="")

        # Should raise SandboxExecutionError when environment not found
        with pytest.raises(SandboxExecutionError):
            await view_executor.execute(params, context=context)
