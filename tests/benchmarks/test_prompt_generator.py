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
