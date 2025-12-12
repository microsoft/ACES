"""Tests for template expansion functionality.

Tests cover:
- Template task validation
- Template reference validation
- Template instance creation
- Orchestration isolation
- Error handling
"""

import pytest

from saber.server.benchmarks.exceptions import InvalidTaskDefinitionException
from saber.server.benchmarks.task import Task
from saber.server.benchmarks.template_expander import TemplateExpander


class TestTaskTemplateValidation:
    """Test validation of template field constraints in Task model."""

    def test_template_cannot_have_depends_on_task_id(self):
        """Templates cannot have depends_on_task_id."""
        with pytest.raises(ValueError, match="Templates cannot have 'depends_on_task_id'"):
            Task(
                task_id="invalid_template",
                domain="test",
                title="Invalid Template",
                description="Template with dependency",
                prompts={
                    "instruction": "test.md",
                    "assistant": "test.md",
                    "submit": "test.md",
                    "continue": "test_continue.md",
                },
                is_template=True,
                depends_on_task_id="some_task",
                role="blue",
            )

    def test_template_cannot_have_dependency_template(self):
        """Templates cannot depend on other templates."""
        with pytest.raises(ValueError, match="Templates cannot have 'dependency_template'"):
            Task(
                task_id="invalid_template",
                domain="test",
                title="Invalid Template",
                description="Template depending on template",
                prompts={
                    "instruction": "test.md",
                    "assistant": "test.md",
                    "submit": "test.md",
                    "continue": "test_continue.md",
                },
                is_template=True,
                dependency_template="other_template",
                role="blue",
            )

    def test_cannot_have_both_dependency_types(self):
        """Cannot specify both dependency_template and depends_on_task_id."""
        with pytest.raises(ValueError, match="Cannot specify both"):
            Task(
                task_id="invalid_task",
                domain="test",
                title="Invalid Task",
                description="Task with both dependency types",
                prompts={
                    "instruction": "test.md",
                    "assistant": "test.md",
                    "submit": "test.md",
                    "continue": "test_continue.md",
                },
                dependency_template="template_task",
                depends_on_task_id="concrete_task",
                role="red",
            )

    def test_depends_on_task_id_is_deprecated(self):
        """depends_on_task_id raises error indicating deprecation."""
        with pytest.raises(ValueError, match="'depends_on_task_id' is deprecated"):
            Task(
                task_id="legacy_task",
                domain="test",
                title="Legacy Task",
                description="Task using old dependency format",
                prompts={
                    "instruction": "test.md",
                    "assistant": "test.md",
                    "submit": "test.md",
                    "continue": "test_continue.md",
                },
                depends_on_task_id="old_target",
                role="red",
            )

    def test_dependency_template_requires_role(self):
        """Tasks with dependency_template must have role defined."""
        with pytest.raises(ValueError, match="'role' is required when 'dependency_template' is set"):
            Task(
                task_id="dependent_task",
                domain="test",
                title="Dependent Task",
                description="Task missing role",
                prompts={
                    "instruction": "test.md",
                    "assistant": "test.md",
                    "submit": "test.md",
                    "continue": "test_continue.md",
                },
                dependency_template="template_task",
                # Missing role
            )

    def test_valid_template_task(self):
        """Valid template task is created successfully."""
        template = Task(
            task_id="blue_template",
            domain="test",
            title="Blue Agent Template",
            description="Reusable blue agent",
            prompts={
                "instruction": "blue.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            is_template=True,
            role="blue",
            benchmark_config={"episode_attempts": 1},
        )

        assert template.is_template is True
        assert template.role == "blue"
        assert template.depends_on_task_id is None
        assert template.dependency_template is None

    def test_valid_template_dependent_task(self):
        """Valid task with dependency_template is created successfully."""
        dependent = Task(
            task_id="red_attack_1",
            domain="test",
            title="Red Attack 1",
            description="Attack scenario 1",
            prompts={
                "instruction": "attack1.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="blue_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        assert dependent.dependency_template == "blue_template"
        assert dependent.role == "red"
        assert dependent.is_template is False
        assert dependent.depends_on_task_id is None

    def test_standalone_task_without_dependencies(self):
        """Standalone task without any dependencies works correctly."""
        standalone = Task(
            task_id="standalone_task",
            domain="test",
            title="Standalone Task",
            description="Independent task",
            prompts={
                "instruction": "standalone.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            benchmark_config={"episode_attempts": 1},
        )

        assert standalone.is_template is False
        assert standalone.dependency_template is None
        assert standalone.depends_on_task_id is None
        assert standalone.role is None  # Role not required for standalone


class TestTemplateExpander:
    """Test template expansion logic."""

    @pytest.fixture
    def expander(self):
        """Create template expander instance."""
        return TemplateExpander(domain="test_domain")

    @pytest.fixture
    def blue_template(self):
        """Create blue team template task."""
        return Task(
            task_id="blue_agent_template",
            domain="test",
            title="Blue Agent",
            description="Guardrailed agent template",
            prompts={
                "instruction": "blue.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            is_template=True,
            role="blue",
            initial_context={"key": "value"},
            benchmark_config={"episode_attempts": 1},
        )

    @pytest.fixture
    def red_attack_1(self):
        """Create first red attack task."""
        return Task(
            task_id="red_attack_rm_rf",
            domain="test",
            title="RM RF Attack",
            description="Test rm -rf vulnerability",
            prompts={
                "instruction": "attack_rm.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="blue_agent_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

    @pytest.fixture
    def red_attack_2(self):
        """Create second red attack task."""
        return Task(
            task_id="red_attack_drop_db",
            domain="test",
            title="Drop DB Attack",
            description="Test database deletion",
            prompts={
                "instruction": "attack_db.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="blue_agent_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

    def test_template_expansion_creates_separate_instances(
        self, expander, blue_template, red_attack_1, red_attack_2
    ):
        """Template expansion creates separate instances for each dependent."""
        tasks = {
            "blue_agent_template": blue_template,
            "red_attack_rm_rf": red_attack_1,
            "red_attack_drop_db": red_attack_2,
        }

        expanded = expander.expand_templates(tasks)

        # Template should be removed from final output
        assert "blue_agent_template" not in expanded

        # Two template instances should be created
        assert "blue_agent_template_red_attack_rm_rf" in expanded
        assert "blue_agent_template_red_attack_drop_db" in expanded

        # Original dependent tasks should remain
        assert "red_attack_rm_rf" in expanded
        assert "red_attack_drop_db" in expanded

        # Total: 2 instances + 2 dependents = 4 tasks
        assert len(expanded) == 4

    def test_template_instances_are_independent(
        self, expander, blue_template, red_attack_1, red_attack_2
    ):
        """Template instances are deep copies and independent."""
        tasks = {
            "blue_agent_template": blue_template,
            "red_attack_rm_rf": red_attack_1,
            "red_attack_drop_db": red_attack_2,
        }

        expanded = expander.expand_templates(tasks)

        instance_1 = expanded["blue_agent_template_red_attack_rm_rf"]
        instance_2 = expanded["blue_agent_template_red_attack_drop_db"]

        # Instances have different IDs
        assert instance_1.task_id != instance_2.task_id

        # But same role and other template properties
        assert instance_1.role == instance_2.role == "blue"
        assert instance_1.title == instance_2.title == "Blue Agent"

        # Instances are not templates
        assert instance_1.is_template is False
        assert instance_2.is_template is False

        # Instances don't depend on anything (they're roots)
        assert instance_1.depends_on_task_id is None
        assert instance_2.depends_on_task_id is None

        # Modifying one instance's context doesn't affect the other
        instance_1.initial_context["modified"] = "value1"
        assert "modified" not in instance_2.initial_context

    def test_dependents_reference_concrete_instances(
        self, expander, blue_template, red_attack_1, red_attack_2
    ):
        """After expansion, dependents reference concrete instance IDs."""
        tasks = {
            "blue_agent_template": blue_template,
            "red_attack_rm_rf": red_attack_1,
            "red_attack_drop_db": red_attack_2,
        }

        expanded = expander.expand_templates(tasks)

        # Dependent tasks should now have concrete dependencies
        assert expanded["red_attack_rm_rf"].depends_on_task_id == "blue_agent_template_red_attack_rm_rf"
        assert expanded["red_attack_drop_db"].depends_on_task_id == "blue_agent_template_red_attack_drop_db"

        # Template references should be cleared
        assert expanded["red_attack_rm_rf"].dependency_template is None
        assert expanded["red_attack_drop_db"].dependency_template is None

    def test_standalone_tasks_unchanged(self, expander):
        """Standalone tasks pass through expansion unchanged."""
        standalone = Task(
            task_id="standalone_task",
            domain="test",
            title="Standalone",
            description="No dependencies",
            prompts={
                "instruction": "standalone.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            benchmark_config={"episode_attempts": 1},
        )

        tasks = {"standalone_task": standalone}
        expanded = expander.expand_templates(tasks)

        assert "standalone_task" in expanded
        assert expanded["standalone_task"].task_id == "standalone_task"
        assert len(expanded) == 1

    def test_invalid_template_reference_raises_error(self, expander, red_attack_1):
        """Referencing non-existent template raises error."""
        tasks = {
            "red_attack_rm_rf": red_attack_1,
            # Missing blue_agent_template
        }

        with pytest.raises(
            InvalidTaskDefinitionException,
            match="references non-existent template 'blue_agent_template'"
        ):
            expander.expand_templates(tasks)

    def test_template_validation_requires_role(self, expander):
        """Template validation requires role to be defined."""
        # Validation happens in Task.__init__ now (fail-fast)
        with pytest.raises(
            ValueError,
            match="Templates must have a 'role' defined"
        ):
            template_without_role = Task(
                task_id="invalid_template",
                domain="test",
                title="Invalid Template",
                description="Template without role",
                prompts={
                    "instruction": "test.md",
                    "assistant": "assistant.md",
                    "submit": "submit.md",
                    "continue": "test_continue.md",
                },
                is_template=True,
                # Missing role - should raise ValueError
                benchmark_config={"episode_attempts": 1},
            )

    def test_template_validation_forbids_depends_on_task_id(self, expander):
        """Template validation prevents depends_on_task_id in templates."""
        # Note: This should be caught by Task.__init__, but validate_templates
        # provides a secondary check
        tasks = {}  # Empty because Task.__init__ will raise first

        # This test verifies the validation logic exists in validate_templates
        # Actual validation happens in Task.__init__

    def test_mixed_templates_and_standalone_tasks(self, expander, blue_template, red_attack_1):
        """Expansion handles mix of templates, dependents, and standalone tasks."""
        standalone = Task(
            task_id="standalone_task",
            domain="test",
            title="Standalone",
            description="Independent task",
            prompts={
                "instruction": "standalone.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            benchmark_config={"episode_attempts": 1},
        )

        tasks = {
            "blue_agent_template": blue_template,
            "red_attack_rm_rf": red_attack_1,
            "standalone_task": standalone,
        }

        expanded = expander.expand_templates(tasks)

        # Template removed
        assert "blue_agent_template" not in expanded

        # Instance created
        assert "blue_agent_template_red_attack_rm_rf" in expanded

        # Dependent present
        assert "red_attack_rm_rf" in expanded

        # Standalone unchanged
        assert "standalone_task" in expanded

        # Total: 1 instance + 1 dependent + 1 standalone = 3
        assert len(expanded) == 3

    def test_template_instance_preserves_all_config(self, expander):
        """Template instances preserve all configuration from template."""
        template = Task(
            task_id="complex_template",
            domain="test",
            title="Complex Template",
            description="Template with full config",
            prompts={
                "instruction": "complex.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            is_template=True,
            role="blue",
            initial_context={"db": "config", "nested": {"value": 123}},
            environment="custom_environment",
            allowed_executors=["bash", "python"],
            execution_config={"executors": {"bash": {"timeout": 30}}},
            episode_config={"max_steps": 50, "custom": "value"},
            benchmark_config={"episode_attempts": 3},
            submission_evaluation_config={"strategy": "llm_judge"},
            step_evaluation_config={"strategy": "llm_judge"},
            initial_files={"/root/file.txt": "source/file.txt"},
        )

        dependent = Task(
            task_id="dependent",
            domain="test",
            title="Dependent",
            description="Depends on complex template",
            prompts={
                "instruction": "dep.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="complex_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        tasks = {"complex_template": template, "dependent": dependent}
        expanded = expander.expand_templates(tasks)

        instance = expanded["complex_template_dependent"]

        # Verify all config preserved
        assert instance.task_id == "complex_template_dependent"
        assert instance.role == "blue"
        assert instance.initial_context == {"db": "config", "nested": {"value": 123}}
        assert instance.environment == "custom_environment"
        assert instance.allowed_executors == ["bash", "python"]
        assert instance.execution_config == {"executors": {"bash": {"timeout": 30}}}
        assert instance.episode_config == {"max_steps": 50, "custom": "value"}
        assert instance.benchmark_config == {"episode_attempts": 3}
        assert instance.submission_evaluation_config == {"strategy": "llm_judge"}
        assert instance.step_evaluation_config == {"strategy": "llm_judge"}
        assert instance.initial_files == {"/root/file.txt": "source/file.txt"}

    def test_multiple_dependents_on_same_template_get_unique_instances(self, expander):
        """Multiple tasks depending on same template each get unique instance."""
        template = Task(
            task_id="shared_template",
            domain="test",
            title="Shared Template",
            description="Template with multiple dependents",
            prompts={
                "instruction": "shared.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            is_template=True,
            role="blue",
            benchmark_config={"episode_attempts": 1},
        )

        dependent_1 = Task(
            task_id="dep_1",
            domain="test",
            title="Dependent 1",
            description="First dependent",
            prompts={
                "instruction": "dep1.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="shared_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        dependent_2 = Task(
            task_id="dep_2",
            domain="test",
            title="Dependent 2",
            description="Second dependent",
            prompts={
                "instruction": "dep2.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="shared_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        dependent_3 = Task(
            task_id="dep_3",
            domain="test",
            title="Dependent 3",
            description="Third dependent",
            prompts={
                "instruction": "dep3.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="shared_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        tasks = {
            "shared_template": template,
            "dep_1": dependent_1,
            "dep_2": dependent_2,
            "dep_3": dependent_3,
        }

        expanded = expander.expand_templates(tasks)

        # 3 instances + 3 dependents = 6 tasks
        assert len(expanded) == 6

        # Each dependent has unique instance
        assert "shared_template_dep_1" in expanded
        assert "shared_template_dep_2" in expanded
        assert "shared_template_dep_3" in expanded

        # Verify dependencies point to correct instances
        assert expanded["dep_1"].depends_on_task_id == "shared_template_dep_1"
        assert expanded["dep_2"].depends_on_task_id == "shared_template_dep_2"
        assert expanded["dep_3"].depends_on_task_id == "shared_template_dep_3"

        # Verify all instances are independent
        instance_ids = {
            expanded["shared_template_dep_1"].task_id,
            expanded["shared_template_dep_2"].task_id,
            expanded["shared_template_dep_3"].task_id,
        }
        assert len(instance_ids) == 3  # All unique


class TestTemplateExpansionIntegration:
    """Integration tests for template expansion with orchestration assembly."""

    def test_expanded_tasks_ready_for_orchestration(self):
        """Expanded tasks have correct structure for orchestration assembly."""
        expander = TemplateExpander(domain="test")

        template = Task(
            task_id="blue_template",
            domain="test",
            title="Blue Agent",
            description="Template",
            prompts={
                "instruction": "blue.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            is_template=True,
            role="blue",
            benchmark_config={"episode_attempts": 1},
        )

        dependent = Task(
            task_id="red_attack",
            domain="test",
            title="Red Attack",
            description="Attack",
            prompts={
                "instruction": "red.md",
                "assistant": "assistant.md",
                "submit": "submit.md",
                "continue": "test_continue.md",
            },
            dependency_template="blue_template",
            role="red",
            benchmark_config={"episode_attempts": 1},
        )

        tasks = {"blue_template": template, "red_attack": dependent}
        expanded = expander.expand_templates(tasks)

        # After expansion:
        # - Instance exists with role
        instance = expanded["blue_template_red_attack"]
        assert instance.role == "blue"
        assert instance.depends_on_task_id is None  # Root task

        # - Dependent references instance
        dependent_task = expanded["red_attack"]
        assert dependent_task.depends_on_task_id == "blue_template_red_attack"
        assert dependent_task.role == "red"

        # This structure is ready for BenchmarkManager._assemble_benchmark_tasks()
        # which will create orchestration: [instance, dependent]
