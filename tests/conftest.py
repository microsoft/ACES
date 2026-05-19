# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Test configuration and utilities for SABER test suite.

This module provides common test fixtures, mock objects, and utilities
that can be shared across multiple test modules.
"""

import logging
import tempfile
from pathlib import Path

import pytest
import yaml

logger = logging.getLogger(__name__)


# Test fixtures
@pytest.fixture
def test_config_path():
    """Path to test configuration file."""
    # Look for config in the appropriate subdirectory based on test location
    test_file_path = Path(__file__).parent

    # Check if we're in a subdirectory and adjust accordingly
    if (test_file_path / "execution" / "config" / "test_command_config.yaml").exists():
        return test_file_path / "execution" / "config" / "test_command_config.yaml"
    elif (test_file_path / "config" / "test_command_config.yaml").exists():
        return test_file_path / "config" / "test_command_config.yaml"
    else:
        # Create a default path for the config file
        return test_file_path / "config" / "test_command_config.yaml"


@pytest.fixture
def test_config_dict(test_config_path):
    """Load test configuration as dictionary."""
    with open(test_config_path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def temp_sandbox_dir():
    """Create a temporary directory for sandbox testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


# Task Framework Test Fixtures


@pytest.fixture
def sample_task_yaml():
    """Sample YAML content for task configuration testing."""
    return """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 3

global_defaults:
  prompts:
    instruction: "malware_family_analysis_prompt.md"
    assistant: "malware_family_analysis_prompt.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "malware_family_analysis"
    title: "Malware Family Classification and Analysis"
    description: "Analyze malware sample to determine family, capabilities, and threat level"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "task_completion"
      scoring:
        max_score: 1.0
    initial_context:
      sample_path: "/data/samples/unknown_sample.exe"
      analysis_timeout: 300
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Analysis"
        description: "Perform static analysis of the malware sample"
        objective: "Extract basic file properties, strings, and structural information"
        completion_conditions: ["file unknown_sample.exe", "strings unknown_sample.exe"]
        depends_on: []
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment"
        objective: "Observe runtime behavior and system interactions"
        completion_conditions: ["sandbox_run unknown_sample.exe"]
        depends_on: ["static_analysis"]
"""


@pytest.fixture
def sample_task_with_environment_yaml():
    """Sample YAML content for task configuration with environment specification."""
    return """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 5

global_defaults:
  prompts:
    instruction: "malware_analysis_prompt.md"
    assistant: "malware_analysis_prompt.md"
    continue: "test_continue.md"
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "malware_analysis_with_env"
    title: "Malware Analysis with Environment"
    description: "Analyze malware sample in a multi-container environment"
    environment: "excytin_db1"
    initial_context:
      sample_path: "/data/samples/unknown_sample.exe"
      analysis_timeout: 300
    subtasks:
      - subtask_id: "static_analysis"
        title: "Static Analysis"
        description: "Perform static analysis of the malware sample"
        objective: "Extract basic file properties, strings, and structural information"
        completion_conditions: ["file unknown_sample.exe", "strings unknown_sample.exe"]
        depends_on: []
      - subtask_id: "dynamic_analysis"
        title: "Dynamic Analysis"
        description: "Execute sample in sandboxed environment with database"
        objective: "Observe runtime behavior and database interactions"
        completion_conditions: ["sandbox_run unknown_sample.exe"]
        depends_on: ["static_analysis"]
"""


@pytest.fixture
def temp_config_dir_helper():
    """Helper fixture to create temporary config directories with hierarchical task structure and prompts directory."""

    def _create_temp_config_dir(tmp_path, yaml_content):
        # Create the config directory
        config_dir = tmp_path / "config"
        config_dir.mkdir()

        # Parse the old YAML content to extract relevant parts
        yaml_data = yaml.safe_load(yaml_content)

        # Create the tasks directory
        tasks_dir = config_dir / "tasks"
        tasks_dir.mkdir()

        default_prompts = {
            "instruction": "instructions/default.md",
            "assistant": "assistants/default.md",
            "continue": "test_continue.md",
        }

        # Create global.yaml with domain and benchmark_config
        global_config = {
            "domain": yaml_data.get("domain"),
            "benchmark_config": yaml_data.get("benchmark_config", {}),
            "global_defaults": yaml_data.get("global_defaults", {}),
            "executors": yaml_data.get("executors", []),
        }

        global_defaults = global_config.setdefault("global_defaults", {})
        global_defaults.setdefault("prompts", default_prompts.copy())

        # Add allowed_executors from top-level to executors if exists
        if "allowed_executors" in yaml_data:
            global_config["executors"] = yaml_data["allowed_executors"]

        # Add permanent_environment if present
        if "permanent_environment" in yaml_data:
            global_config["permanent_environment"] = yaml_data["permanent_environment"]

        global_file = tasks_dir / "global.yaml"
        global_file.write_text(yaml.dump(global_config, default_flow_style=False))

        # Create task files from the tasks list
        if "tasks" in yaml_data and yaml_data["tasks"]:
            # For each task, create a separate file or group them
            for i, task in enumerate(yaml_data["tasks"]):
                task_file = tasks_dir / f"task_{i + 1}.yaml"
                task_copy = dict(task)
                template = task_copy.pop("prompt_template_file", None)
                if template:
                    task_copy.setdefault(
                        "prompts",
                        {
                            "instruction": template,
                            "assistant": template,
                        },
                    )
                else:
                    task_copy.setdefault("prompts", default_prompts.copy())

                task_data = {"tasks": [task_copy]}
                task_file.write_text(yaml.dump(task_data, default_flow_style=False))
        else:
            # If no tasks, create an empty task file to satisfy the loader
            task_file = tasks_dir / "empty_tasks.yaml"
            task_data = {"tasks": []}
            task_file.write_text(yaml.dump(task_data, default_flow_style=False))

        # Create the prompts directory
        prompts_dir = config_dir / "prompts"
        prompts_dir.mkdir()

        # Create some sample prompt template files (legacy single-prompt format)
        for template_name in [
            "malware_family_analysis_prompt.md",
            "malware_analysis_prompt.md",
            "test_task_prompt.md",
            "test_continue.md",
        ]:
            template_file = prompts_dir / template_name
            template_file.write_text("# Sample Template\n\nThis is a sample prompt template for testing.")

        # Create multi-prompt directory structure
        instructions_dir = prompts_dir / "instructions"
        instructions_dir.mkdir(exist_ok=True)
        assistants_dir = prompts_dir / "assistants"
        assistants_dir.mkdir(exist_ok=True)

        # Create multi-prompt template files
        multi_prompt_templates = [
            (
                "instructions/default.md",
                "# Default Instruction\n\nThis is the default instruction template for testing.",
            ),
            (
                "assistants/default.md",
                "# Default Assistant\n\nThis is the default assistant template for testing.",
            ),
            (
                "instructions/security_analysis_instruction.md",
                "# Security Analysis Instruction\n\nYou are a security analyst. Your task is to...",
            ),
            (
                "assistants/security_analysis_assistant.md",
                "# Security Analysis Assistant\n\nAs an assistant, help the agent by...",
            ),
        ]

        for template_path, content in multi_prompt_templates:
            template_file = prompts_dir / template_path
            template_file.write_text(content)

        # Create judge templates directory and files
        judge_dir = prompts_dir / "judge"
        judge_dir.mkdir(exist_ok=True)
        for judge_template_name in ["system.md", "user.md"]:
            judge_template_file = judge_dir / judge_template_name
            judge_template_file.write_text("# Sample Judge Template\n\nThis is a sample judge template for testing.")

        return str(config_dir)

    return _create_temp_config_dir


@pytest.fixture
def temp_config_dir(tmp_path, sample_task_yaml):
    """Create a temporary directory with hierarchical task configuration and return the directory path."""
    # Parse the YAML content to extract relevant parts
    yaml_data = yaml.safe_load(sample_task_yaml)

    # Create the tasks directory
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir()

    # Create global.yaml with domain and benchmark_config
    global_config = {
        "domain": yaml_data.get("domain"),
        "benchmark_config": yaml_data.get("benchmark_config", {}),
        "global_defaults": yaml_data.get("global_defaults", {}),
        "executors": yaml_data.get("executors", []),
    }

    # Add allowed_executors from top-level to executors if exists
    if "allowed_executors" in yaml_data:
        global_config["executors"] = yaml_data["allowed_executors"]

    # Add permanent_environment if present
    if "permanent_environment" in yaml_data:
        global_config["permanent_environment"] = yaml_data["permanent_environment"]

    global_file = tasks_dir / "global.yaml"
    global_file.write_text(yaml.dump(global_config, default_flow_style=False))

    # Create task files from the tasks list
    if "tasks" in yaml_data and yaml_data["tasks"]:
        # For each task, create a separate file or group them
        for i, task in enumerate(yaml_data["tasks"]):
            task_file = tasks_dir / f"task_{i + 1}.yaml"
            task_data = {"tasks": [task]}
            task_file.write_text(yaml.dump(task_data, default_flow_style=False))
    else:
        # If no tasks, create an empty task file to satisfy the loader
        task_file = tasks_dir / "empty_tasks.yaml"
        task_data = {"tasks": []}
        task_file.write_text(yaml.dump(task_data, default_flow_style=False))

    # Create the prompts directory
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()

    # Create some sample prompt template files
    for template_name in [
        "malware_family_analysis_prompt.md",
        "malware_analysis_prompt.md",
        "test_task_prompt.md",
        "test_continue.md",
    ]:
        template_file = prompts_dir / template_name
        template_file.write_text("# Sample Template\n\nThis is a sample prompt template for testing.")

    # Create judge templates directory and files
    judge_dir = prompts_dir / "judge"
    judge_dir.mkdir(exist_ok=True)
    for judge_template_name in ["system.md", "user.md"]:
        judge_template_file = judge_dir / judge_template_name
        judge_template_file.write_text("# Sample Judge Template\n\nThis is a sample judge template for testing.")

    return str(tmp_path)  # Return directory path


@pytest.fixture
def sample_subtask_data():
    """Sample subtask data for testing."""
    return {
        "subtask_id": "test_subtask",
        "task_id": "test_task",
        "title": "Test SubTask",
        "description": "A test subtask for unit testing",
        "objective": "Complete the test objectives",
        "completion_conditions": ["test_command", "another_command"],
        "depends_on": ["prerequisite_subtask"],
    }


@pytest.fixture(scope="function")
def docker_cleanup():
    """
    Fixture to ensure Docker containers are cleaned up after tests.

    This fixture tracks execution managers and session IDs used during tests
    and ensures proper cleanup even if tests fail.
    """
    execution_managers = []
    session_ids = []

    def register_execution_manager(execution_manager, session_id=None):
        """Register an execution manager and optional session ID for cleanup."""
        execution_managers.append(execution_manager)
        if session_id:
            session_ids.append((execution_manager, session_id))

    yield register_execution_manager

    # Cleanup after test
    for execution_manager, session_id in session_ids:
        try:
            execution_manager.cleanup_session(session_id)
            logger.debug(f"Cleaned up session {session_id}")
        except Exception as e:
            logger.warning(f"Failed to cleanup session {session_id}: {e}")

    for execution_manager in execution_managers:
        try:
            # Cleanup all sessions if the manager has that capability
            if hasattr(execution_manager, "_sandbox_manager"):
                execution_manager._sandbox_manager.cleanup_all_sessions()
                logger.debug("Cleaned up all sessions from execution manager")
        except Exception as e:
            logger.warning(f"Failed to cleanup execution manager: {e}")


# Additional fixtures for benchmark configuration testing


@pytest.fixture
def sample_task_with_task_level_benchmark_config_yaml():
    """Sample YAML content with task-level benchmark configuration override."""
    return """
domain: "webapp_pentest"

benchmark_config:
  episode_attempts: 3

tasks:
  - task_id: "xss_detection"
    title: "XSS Detection Task"
    description: "Detect cross-site scripting vulnerabilities"
    benchmark_config:
      episode_attempts: 10
    subtasks:
      - subtask_id: "input_analysis"
        title: "Input Analysis"
        description: "Analyze input fields for XSS vulnerabilities"
        objective: "Identify potential XSS injection points"
  - task_id: "sql_injection"
    title: "SQL Injection Task"
    description: "Detect SQL injection vulnerabilities"
    subtasks:
      - subtask_id: "parameter_analysis"
        title: "Parameter Analysis"
        description: "Analyze parameters for SQL injection"
        objective: "Identify SQL injection vulnerabilities"
"""


@pytest.fixture
def sample_yaml_missing_benchmark_config():
    """Sample YAML content missing benchmark_config section (should fail)."""
    return """
domain: "malware_classification"

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task without benchmark config"
    prompt_template_file: "test_task_prompt.md"
    subtasks: []
"""


@pytest.fixture
def sample_yaml_missing_episode_attempts():
    """Sample YAML content missing episode_attempts (should fail)."""
    return """
domain: "malware_classification"

benchmark_config:
  other_setting: true

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task without episode_attempts"
    prompt_template_file: "test_task_prompt.md"
    subtasks: []
"""


@pytest.fixture
def sample_yaml_invalid_episode_attempts():
    """Sample YAML content with invalid episode_attempts (should fail)."""
    return """
domain: "malware_classification"

benchmark_config:
  episode_attempts: 0

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "test_task"
    title: "Test Task"
    description: "A test task with invalid episode_attempts"
    prompt_template_file: "test_task_prompt.md"
    subtasks: []
"""


@pytest.fixture
def sample_multi_prompt_task_yaml():
    """Sample YAML content for multi-prompt task configuration testing."""
    return """
domain: "cybersecurity"

benchmark_config:
  episode_attempts: 2

global_defaults:
  execution_config:
    executors:
      bash_executor:
        timeout: 300
      python_executor:
        timeout: 300
  episode_config:
    max_steps: 50

executors:
  - bash_executor
  - python_executor

tasks:
  - task_id: "multi_prompt_security_task"
    title: "Multi-Prompt Security Analysis"
    description: "A security task with separate instruction and assistant prompts"
    prompts:
      instruction: "instructions/security_analysis_instruction.md"
      assistant: "assistants/security_analysis_assistant.md"
      continue: "test_continue.md"
    execution_config:
      executors:
        bash_executor:
          timeout: 300
        python_executor:
          timeout: 300
    submission_evaluation_config:
      strategy: "static"
      criteria:
        expected_answers:
          - "vulnerability_found"
      scoring:
        max_score: 1.0
    initial_context:
      target_url: "http://example.com"
      scan_timeout: 300
    subtasks:
      - subtask_id: "reconnaissance"
        title: "Reconnaissance"
        description: "Gather information about the target"
        objective: "Identify potential attack vectors"
        completion_conditions: ["nmap_scan", "directory_enumeration"]
        depends_on: []
      - subtask_id: "vulnerability_assessment"
        title: "Vulnerability Assessment"
        description: "Identify security vulnerabilities"
        objective: "Find exploitable vulnerabilities"
        completion_conditions: ["vulnerability_scan", "manual_testing"]
        depends_on: ["reconnaissance"]
"""
