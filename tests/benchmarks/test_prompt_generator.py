"""Unit tests for PromptGenerator service (Agent prompts only).

CLIENT-SIDE EVALUATION: Judge prompt generation moved to client side.
This file now only tests agent prompt generation and template validation.
"""

import tempfile
from pathlib import Path

import pytest

from saber.server.benchmarks.prompt_generator import (
    PromptContext,
    PromptGenerationError,
    PromptGenerator,
)
from saber.server.benchmarks.subtask import SubTask
from saber.server.benchmarks.task import Task


class TestPromptContext:
    """Test PromptContext data container."""

    def test_prompt_context_creation(self):
        """Test creating PromptContext with all fields."""
        context = PromptContext(
            domain="webapp_pentest",
            task_id="xss_0_flag_capture",
            task_title="XSS_0 Vulnerability Exploitation",
            task_description="Test description",
            timeout_seconds=30,
            max_steps=5,
            environment="xss_0_ctf",
            subtasks=[{"title": "Test", "description": "Desc", "objective": "Obj"}],
            allowed_executors=["bash", "python"]
        )

        assert context.domain == "webapp_pentest"
        assert context.task_id == "xss_0_flag_capture"
        assert context.timeout_seconds == 30
        assert len(context.subtasks) == 1

    def test_prompt_context_to_dict(self):
        """Test converting PromptContext to dictionary."""
        context = PromptContext(
            domain="test_domain",
            task_id="test_task",
            task_title="Test Task",
            task_description="Test description",
            timeout_seconds=60,
            max_steps=10,
            environment="test_env",
            subtasks=[],
            allowed_executors=["bash"]
        )

        context_dict = context.to_dict()
        assert isinstance(context_dict, dict)
        assert context_dict["domain"] == "test_domain"
        assert context_dict["timeout_seconds"] == 60
        assert context_dict["subtasks"] == []

    def test_prompt_context_with_initial_files(self):
        """Test PromptContext with initial_files field."""
        initial_files = {
            "/root/pom.xml": "/app/data/resources/pom.xml",
            "/root/src/": "/app/data/resources/src/"
        }

        context = PromptContext(
            domain="cybench",
            task_id="labyrinth_linguist",
            task_title="Labyrinth Linguist Challenge",
            task_description="Test description",
            timeout_seconds=60,
            max_steps=10,
            environment="test_env",
            subtasks=[],
            allowed_executors=["bash"],
            initial_files=initial_files
        )

        assert context.initial_files == initial_files

        context_dict = context.to_dict()
        assert "initial_files" in context_dict
        assert context_dict["initial_files"] == initial_files
        assert context_dict["initial_files"]["/root/pom.xml"] == "/app/data/resources/pom.xml"

    def test_prompt_context_without_initial_files(self):
        """Test PromptContext without initial_files (defaults to None)."""
        context = PromptContext(
            domain="test_domain",
            task_id="test_task",
            task_title="Test Task",
            task_description="Test description",
            timeout_seconds=60,
            max_steps=10,
            environment="test_env",
            subtasks=[],
            allowed_executors=["bash"]
        )

        assert context.initial_files is None

        context_dict = context.to_dict()
        assert "initial_files" not in context_dict  # Should not be included when None


class TestPromptGenerator:
    """Test PromptGenerator service for agent prompts."""

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir) / "prompts"
            prompts_dir.mkdir()

            # Create a basic template
            basic_template = prompts_dir / "basic_prompt.md"
            basic_template.write_text("Test template for {{ task_title }}")

            yield prompts_dir

    @pytest.fixture
    def sample_task(self):
        """Create sample task for testing."""
        subtask = SubTask(
            subtask_id="test_subtask",
            task_id="test_task",
            title="Test Subtask",
            description="Test subtask description",
            objective="Test objective"
        )

        return Task(
            task_id="test_task",
            domain="test_domain",
            title="Test Task",
            description="Test task description",
            prompts={
                "instruction": "basic_prompt.md",
                "assistant": "basic_prompt.md",
                "submit": "basic_prompt.md",
                "continue": "basic_prompt.md"
            },
            subtasks=[subtask],
            execution_config={"timeout": 30},
            episode_config={"max_steps": 5},
            allowed_executors=["bash", "python"]
        )

    def test_init_with_valid_directory(self, temp_prompts_dir):
        """Test PromptGenerator initialization with valid directory."""
        generator = PromptGenerator(str(temp_prompts_dir))
        assert generator.prompts_dir == temp_prompts_dir
        assert generator.jinja_env is not None
    def test_init_with_custom_domain_root(self, temp_prompts_dir):
        """Test PromptGenerator initialization with custom domain_root."""
        with tempfile.TemporaryDirectory() as domain_root:
            generator = PromptGenerator(str(temp_prompts_dir), domain_root=domain_root)
            assert generator.prompts_dir == temp_prompts_dir
            assert generator.domain_root == Path(domain_root)

    def test_init_default_domain_root(self, temp_prompts_dir):
        """Test PromptGenerator default domain_root is prompts_dir's grandparent."""
        generator = PromptGenerator(str(temp_prompts_dir))
        # prompts_dir is temp_dir/prompts, so domain_root should be temp_dir's parent
        expected_root = temp_prompts_dir.parent.parent
        assert generator.domain_root == expected_root


class TestResolveFileReferences:
    """Test file reference resolution in initial_context."""

    @pytest.fixture
    def temp_domain_structure(self):
        """Create temporary domain structure with prompts and data directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            domain_root = Path(temp_dir)

            # Create config/prompts structure
            prompts_dir = domain_root / "config" / "prompts"
            prompts_dir.mkdir(parents=True)
            (prompts_dir / "basic_prompt.md").write_text("Test {{ initial_context.threat_intel_content }}")

            # Create server/data structure
            data_dir = domain_root / "server" / "data" / "threat_intel"
            data_dir.mkdir(parents=True)
            (data_dir / "advisory.md").write_text("# Threat Advisory\n\nThis is a test advisory.")

            yield {
                "domain_root": domain_root,
                "prompts_dir": prompts_dir,
                "data_dir": data_dir,
            }

    def test_resolve_file_references_loads_content(self, temp_domain_structure):
        """Test that _file keys are resolved to _content with file contents."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "threat_intel_file": "server/data/threat_intel/advisory.md",
            "other_key": "other_value",
        }

        result = generator._resolve_file_references(initial_context)

        assert "threat_intel_content" in result
        assert result["threat_intel_content"] == "# Threat Advisory\n\nThis is a test advisory."
        assert result["threat_intel_file"] == "server/data/threat_intel/advisory.md"
        assert result["other_key"] == "other_value"

    def test_resolve_file_references_preserves_original(self, temp_domain_structure):
        """Test that original initial_context is not mutated."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "threat_intel_file": "server/data/threat_intel/advisory.md",
        }
        original_keys = set(initial_context.keys())

        generator._resolve_file_references(initial_context)

        assert set(initial_context.keys()) == original_keys

    def test_resolve_file_references_none_context(self, temp_domain_structure):
        """Test handling None initial_context."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        result = generator._resolve_file_references(None)
        assert result is None

    def test_resolve_file_references_empty_context(self, temp_domain_structure):
        """Test handling empty initial_context."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        result = generator._resolve_file_references({})
        assert result == {}

    def test_resolve_file_references_no_file_keys(self, temp_domain_structure):
        """Test that context without _file keys is returned unchanged."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "kusto_endpoint": "http://localhost:8080",
            "available_tables": ["ContainerLogs", "AADSignInLogs"],
        }

        result = generator._resolve_file_references(initial_context)
        assert result == initial_context

    def test_resolve_file_references_missing_file(self, temp_domain_structure):
        """Test error when referenced file does not exist."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "missing_file": "server/data/nonexistent.md",
        }

        with pytest.raises(PromptGenerationError) as exc_info:
            generator._resolve_file_references(initial_context)

        assert "File not found" in str(exc_info.value)
        assert "missing_file" in str(exc_info.value)

    def test_resolve_file_references_path_traversal_dotdot(self, temp_domain_structure):
        """Test error when path contains '..' (path traversal attempt)."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "malicious_file": "../../../etc/passwd",
        }

        with pytest.raises(PromptGenerationError) as exc_info:
            generator._resolve_file_references(initial_context)

        assert "Unsafe file path" in str(exc_info.value)

    def test_resolve_file_references_absolute_path(self, temp_domain_structure):
        """Test error when path is absolute."""
        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "absolute_file": "/etc/passwd",
        }

        with pytest.raises(PromptGenerationError) as exc_info:
            generator._resolve_file_references(initial_context)

        assert "Unsafe file path" in str(exc_info.value)

    def test_resolve_file_references_multiple_files(self, temp_domain_structure):
        """Test resolving multiple file references."""
        # Create second file
        second_file = temp_domain_structure["data_dir"] / "second.md"
        second_file.write_text("Second file content")

        generator = PromptGenerator(
            str(temp_domain_structure["prompts_dir"]),
            domain_root=str(temp_domain_structure["domain_root"]),
        )

        initial_context = {
            "threat_intel_file": "server/data/threat_intel/advisory.md",
            "secondary_file": "server/data/threat_intel/second.md",
            "other_key": "value",
        }

        result = generator._resolve_file_references(initial_context)

        assert "threat_intel_content" in result
        assert "secondary_content" in result
        assert result["threat_intel_content"] == "# Threat Advisory\n\nThis is a test advisory."
        assert result["secondary_content"] == "Second file content"
        assert result["other_key"] == "value"
