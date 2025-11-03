"""Integration tests for multi-domain template validation and prompt generation."""

import pytest
import tempfile
from pathlib import Path

from saber.server.benchmarks.benchmark_manager import BenchmarkManager
from saber.server.benchmarks.prompt_generator import TemplateValidationError


class TestMultiDomainTemplateValidation:
    """Test template validation and prompt generation across multiple domains."""

    @pytest.mark.skip(reason="Integration test requires full repo structure with domains at repo root")
    def test_all_domains_validate_successfully(self):
        """Test that all production domains validate their templates successfully."""

        # Test cybench domain (excytin_demo removed as it doesn't exist in this repo)
        try:
            bm_cybench = BenchmarkManager('cybench', 'domains/cybench/server/config')
            assert len(bm_cybench.tasks) > 0
            print(f"cybench: SUCCESS ({len(bm_cybench.tasks)} tasks)")
        except Exception as e:
            pytest.fail(f"cybench domain validation failed: {e}")

    @pytest.mark.skip(reason="Integration test requires full repo structure with domains at repo root")
    def test_prompt_generation_all_domains(self):
        """Test prompt generation works for all tasks in all domains."""

        domains = [
            ('cybench', 'domains/cybench/server/config'),
        ]

        total_prompts_generated = 0

        for domain_name, config_path in domains:
            bm = BenchmarkManager(domain_name, config_path)

            for task_id in bm.tasks.keys():
                prompt = bm.get_task_prompt(task_id)
                assert isinstance(prompt, str)
                assert len(prompt) > 100  # Reasonable minimum prompt length
                total_prompts_generated += 1

        print(f"Successfully generated {total_prompts_generated} prompts across {len(domains)} domains")
        assert total_prompts_generated > 0

    @pytest.mark.skip(reason="Integration test requires full repo structure with domains at repo root")
    def test_template_inheritance_validation(self):
        """Test that template inheritance (extends) works correctly."""

        bm = BenchmarkManager('cybench', 'domains/cybench/server/config')

        # Test that templates validate correctly for cybench
        task_ids = list(bm.tasks.keys())
        if task_ids:
            # Test the first available task template
            first_task = bm.tasks[task_ids[0]]
            try:
                prompt = bm.get_task_prompt(task_ids[0])
                assert isinstance(prompt, str)
                assert len(prompt) > 0
                print("Template inheritance validation: SUCCESS")
            except Exception as e:
                pytest.fail(f"Template validation failed: {e}")
        else:
            pytest.skip("No tasks available in cybench domain")

    def test_missing_template_fails_fast(self):
        """Test that missing templates cause startup failure (fail-fast behavior)."""

        # Create a temporary domain with missing template
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "config"
            temp_config.mkdir()

            # Create tasks directory and files for hierarchical structure
            tasks_dir = temp_config / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml
            global_yaml = tasks_dir / "global.yaml"
            global_yaml.write_text("""
domain: "test_domain"

global_defaults:
  prompts:
    instruction: "missing_template.md"
    assistant: "missing_template.md"
    submit: "missing_template.md"
  execution_config:
    executors:
      bash:
        timeout: 30
  episode_config:
    max_steps: 5
  benchmark_config:
    episode_attempts: 1

executors:
  - bash
""")

            # Create task file with missing template
            task_yaml = tasks_dir / "test_task.yaml"
            task_yaml.write_text("""
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "Test description"
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers: ["test_flag"]
      scoring:
        max_score: 1.0
""")

            # Create prompts directory but no template file
            prompts_dir = temp_config / "prompts"
            prompts_dir.mkdir()

            # NOTE: The missing_template.md file is intentionally not created
            # to test template validation failure

            # This should fail fast during initialization
            with pytest.raises(TemplateValidationError) as exc_info:
                BenchmarkManager('test_domain', str(temp_config))

            assert "missing dependencies" in str(exc_info.value)
            print("Missing template fail-fast behavior: SUCCESS")

    def test_missing_required_config_fails_fast(self):
        """Test that missing required configuration causes validation failure."""

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_config = Path(temp_dir) / "config"
            temp_config.mkdir()

            # Create tasks directory for hierarchical structure
            tasks_dir = temp_config / "tasks"
            tasks_dir.mkdir()

            # Create global.yaml with missing required config
            global_yaml = tasks_dir / "global.yaml"
            global_yaml.write_text("""
domain: "test_domain"

# Missing global_defaults - should cause failure
""")

            # Create task file
            task_yaml = tasks_dir / "test_task.yaml"
            task_yaml.write_text("""
tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "Test description"
    prompt_template_file: "test_template.md"
""")

            # Create prompts directory with template
            prompts_dir = temp_config / "prompts"
            prompts_dir.mkdir()

            template_file = prompts_dir / "test_template.md"
            template_file.write_text("Test template: {{ task_title }}")

            # This should fail due to missing required configuration
            with pytest.raises(Exception):  # Could be various config validation errors
                BenchmarkManager('test_domain', str(temp_config))

            print("Missing required config fail-fast behavior: SUCCESS")

    @pytest.mark.skip(reason="Integration test requires full repo structure with domains at repo root")
    def test_shared_partials_work_across_domains(self):
        """Test that shared partials (includes) work across different domains."""

        domains = [
            ('cybench', 'domains/cybench/server/config'),
        ]

        for domain_name, config_path in domains:
            bm = BenchmarkManager(domain_name, config_path)

            # Generate a prompt to ensure includes work
            first_task_id = list(bm.tasks.keys())[0]
            prompt = bm.get_task_prompt(first_task_id)

            # Check that shared content is included
            assert "EXECUTION GUIDELINES" in prompt or "CONSTRAINTS" in prompt

        print("Shared partials validation across domains: SUCCESS")
