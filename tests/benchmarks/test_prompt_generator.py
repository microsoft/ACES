"""Unit tests for PromptGenerator service (Agent prompts only).

CLIENT-SIDE EVALUATION: Judge prompt generation moved to client side.
This file now only tests agent prompt generation and template validation.
"""

import pytest
import tempfile
from pathlib import Path

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
                "submit": "basic_prompt.md"
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
