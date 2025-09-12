"""Unit tests for PromptGenerator service."""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from saber.server.benchmarks.prompt_generator import (
    PromptGenerator,
    PromptContext,
    PromptGenerationError,
    TemplateValidationError,
    PromptContextError,
)
from saber.server.benchmarks.task import Task
from saber.server.benchmarks.subtask import SubTask


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


class TestPromptGenerator:
    """Test PromptGenerator service."""

    @pytest.fixture
    def temp_prompts_dir(self):
        """Create temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            prompts_dir = Path(temp_dir) / "prompts"
            prompts_dir.mkdir()

            # Create a basic template
            basic_template = prompts_dir / "basic_prompt.md"
            basic_template.write_text("""
You are an agent for {{ domain }}.

TASK: {{ task_title }}
DESCRIPTION: {{ task_description }}
TIMEOUT: {{ timeout_seconds }} seconds
MAX STEPS: {{ max_steps }}

{% if subtasks %}
SUBTASKS:
{% for subtask in subtasks %}
- {{ subtask.title }}: {{ subtask.description }}
{% endfor %}
{% endif %}

EXECUTORS: {{ allowed_executors | join(', ') }}
""".strip())

            # Create template with includes
            shared_dir = prompts_dir / "shared"
            shared_dir.mkdir()

            shared_template = shared_dir / "common_guidelines.md"
            shared_template.write_text("COMMON GUIDELINES: Follow security protocols.")

            include_template = prompts_dir / "with_include.md"
            include_template.write_text("""
Task: {{ task_title }}
{% include 'shared/common_guidelines.md' %}
""".strip())

            # Create template with syntax error
            error_template = prompts_dir / "syntax_error.md"
            error_template.write_text("{{ invalid_syntax")

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
            prompt_template_file="basic_prompt.md",
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

    def test_init_with_missing_directory(self):
        """Test PromptGenerator initialization fails with missing directory."""
        with pytest.raises(TemplateValidationError) as exc_info:
            PromptGenerator("/nonexistent/directory")

        assert "Prompts directory does not exist" in str(exc_info.value)

    def test_init_with_file_instead_of_directory(self, temp_prompts_dir):
        """Test PromptGenerator initialization fails when path is a file."""
        file_path = temp_prompts_dir / "not_a_directory.txt"
        file_path.write_text("test")

        with pytest.raises(TemplateValidationError) as exc_info:
            PromptGenerator(str(file_path))

        assert "Prompts path is not a directory" in str(exc_info.value)

    def test_render_prompt_for_task_success(self, temp_prompts_dir, sample_task):
        """Test successful prompt rendering."""
        generator = PromptGenerator(str(temp_prompts_dir))
        rendered = generator.render_prompt_for_task(sample_task)

        assert "You are an agent for test_domain" in rendered
        assert "TASK: Test Task" in rendered
        assert "TIMEOUT: 30 seconds" in rendered
        assert "MAX STEPS: 5" in rendered
        assert "Test Subtask: Test subtask description" in rendered
        assert "EXECUTORS: bash, python" in rendered

    def test_render_prompt_with_includes(self, temp_prompts_dir):
        """Test prompt rendering with template includes."""
        task = Task(
            task_id="include_test",
            domain="test_domain",
            title="Include Test",
            description="Test with includes",
            prompt_template_file="with_include.md",
            execution_config={"timeout": 10},
            episode_config={"max_steps": 3},
            allowed_executors=["bash"],
        )

        generator = PromptGenerator(str(temp_prompts_dir))
        rendered = generator.render_prompt_for_task(task)

        assert "Task: Include Test" in rendered
        assert "COMMON GUIDELINES: Follow security protocols" in rendered

    def test_render_prompt_missing_template_file(self, temp_prompts_dir):
        """Test prompt rendering fails with missing template file."""
        task = Task(
            task_id="missing_test",
            domain="test_domain",
            title="Missing Test",
            description="Test missing template",
            prompt_template_file="nonexistent.md",
            execution_config={"timeout": 10},
            episode_config={"max_steps": 3},
            allowed_executors=["bash"],
        )

        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(TemplateValidationError) as exc_info:
            generator.render_prompt_for_task(task)

        # Updated assertion to reflect new error message formatting (lowercase start)
        assert "template file not found" in str(exc_info.value)
        assert "nonexistent.md" in str(exc_info.value)

    def test_render_prompt_missing_template_file_field(self, temp_prompts_dir):
        """Test prompt rendering fails when task has no template file specified."""
        task = Task(
            task_id="no_template",
            domain="test_domain",
            title="No Template",
            description="Test no template",
            prompt_template_file="",  # Empty string
            execution_config={"timeout": 10},
            episode_config={"max_steps": 3},
            allowed_executors=["bash"],
        )

        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(PromptGenerationError) as exc_info:
            generator.render_prompt_for_task(task)

        assert "missing required prompt_template_file" in str(exc_info.value)

    def test_render_prompt_template_syntax_error(self, temp_prompts_dir):
        """Test prompt rendering fails with template syntax error."""
        task = Task(
            task_id="syntax_error_test",
            domain="test_domain",
            title="Syntax Error Test",
            description="Test syntax error",
            prompt_template_file="syntax_error.md",
            execution_config={"timeout": 10},
            episode_config={"max_steps": 3},
            allowed_executors=["bash"],
        )

        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(PromptGenerationError) as exc_info:
            generator.render_prompt_for_task(task)

        assert "Template rendering failed" in str(exc_info.value)

    def test_validate_template_success(self, temp_prompts_dir):
        """Test successful template validation."""
        generator = PromptGenerator(str(temp_prompts_dir))
        result = generator.validate_template("basic_prompt.md")
        assert result is True

    def test_validate_template_missing_file(self, temp_prompts_dir):
        """Test template validation fails for missing file."""
        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(TemplateValidationError) as exc_info:
            generator.validate_template("missing.md")

        assert "missing dependencies" in str(exc_info.value)
        assert "missing.md" in str(exc_info.value)

    def test_validate_template_syntax_error(self, temp_prompts_dir):
        """Test template validation fails for syntax error."""
        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(TemplateValidationError) as exc_info:
            generator.validate_template("syntax_error.md")

        assert "Template syntax error" in str(exc_info.value)

    def test_validate_all_task_templates_success(self, temp_prompts_dir):
        """Test validation of all task templates succeeds."""
        tasks = [
            Task(
                task_id="task1",
                domain="test",
                title="Task 1",
                description="Description 1",
                prompt_template_file="basic_prompt.md"
            ),
            Task(
                task_id="task2",
                domain="test",
                title="Task 2",
                description="Description 2",
                prompt_template_file="with_include.md"
            )
        ]

        generator = PromptGenerator(str(temp_prompts_dir))
        # Should not raise exception
        generator.validate_all_task_templates(tasks)

    def test_validate_all_task_templates_missing_field(self, temp_prompts_dir):
        """Test validation fails when task missing template file field."""
        tasks = [
            Task(
                task_id="good_task",
                domain="test",
                title="Good Task",
                description="Good description",
                prompt_template_file="basic_prompt.md"
            ),
            Task(
                task_id="bad_task",
                domain="test",
                title="Bad Task",
                description="Bad description",
                prompt_template_file=""  # Missing template
            )
        ]

        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(TemplateValidationError) as exc_info:
            generator.validate_all_task_templates(tasks)

        error_msg = str(exc_info.value)
        assert "Template validation failed" in error_msg
        assert "bad_task" in error_msg
        assert "missing prompt_template_file" in error_msg

    def test_validate_all_task_templates_missing_files(self, temp_prompts_dir):
        """Test validation fails when template files don't exist."""
        tasks = [
            Task(
                task_id="missing_task",
                domain="test",
                title="Missing Task",
                description="Missing description",
                prompt_template_file="missing.md"
            )
        ]

        generator = PromptGenerator(str(temp_prompts_dir))

        with pytest.raises(TemplateValidationError) as exc_info:
            generator.validate_all_task_templates(tasks)

        error_msg = str(exc_info.value)
        assert "Template validation failed" in error_msg
        assert "missing.md" in error_msg

    def test_build_context_from_task_complete(self, sample_task):
        """Test building context from task with all fields."""
        generator = PromptGenerator("/tmp")  # Don't need real dir for this test
        context = generator._build_context_from_task(sample_task)

        assert context.domain == "test_domain"
        assert context.task_id == "test_task"
        assert context.task_title == "Test Task"
        assert context.task_description == "Test task description"
        assert context.timeout_seconds == 30
        assert context.max_steps == 5
        assert context.environment == "default"  # None environment becomes "default"
        assert len(context.subtasks) == 1
        assert context.subtasks[0]["title"] == "Test Subtask"
        assert context.allowed_executors == ["bash", "python"]

    def test_build_context_from_task_missing_required_config(self):
        """Test that missing required config fields raises PromptContextError (fail-fast)."""
        task = Task(
            task_id="minimal_task",
            domain="minimal_domain",
            title="Minimal Task",
            description="Minimal description",
            prompt_template_file="basic.md"
        )

        generator = PromptGenerator("/tmp")
        with pytest.raises(PromptContextError) as exc_info:
            generator._build_context_from_task(task)
        msg = str(exc_info.value)
        assert "execution_config.timeout" in msg
        assert "episode_config.max_steps" in msg
        assert "allowed_executors" in msg
