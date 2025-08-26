"""
Unit tests for PolicyManager.
Tests policy document generation and task-specific configuration.
"""

from unittest.mock import MagicMock

import pytest

from saber.server.policy.policy_manager import PolicyDocument, PolicyManager


class TestPolicyManager:
    """Test cases for PolicyManager functionality."""

    def test_policy_manager_init(self):
        """Test PolicyManager initialization."""
        manager = PolicyManager("test_domain")
        assert manager.domain_name == "test_domain"
        assert manager._timeout_seconds == 60  # Default timeout
        assert manager._current_policy is None

    def test_configure_for_task_with_timeout(self):
        """Test configuring PolicyManager with task that has timeout."""
        manager = PolicyManager("test_domain")

        # Mock task with execution config containing timeout
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 120, "other_config": "value"}

        # Configure for task
        manager.configure_for_task("session_123", mock_task)

        # Verify timeout was extracted and set
        assert manager._timeout_seconds == 120
        assert manager._current_policy is not None
        assert isinstance(manager._current_policy, PolicyDocument)

        # Verify the timeout appears in the policy prompt
        assert "120 second command timeout" in manager._current_policy.prompt

    def test_configure_for_task_without_timeout(self):
        """Test configuring PolicyManager with task that has no timeout."""
        manager = PolicyManager("test_domain")

        # Mock task without execution config
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {}

        # Configure for task
        manager.configure_for_task("session_123", mock_task)

        # Verify default timeout was used
        assert manager._timeout_seconds == 60
        assert manager._current_policy is not None

        # Verify the default timeout appears in the policy prompt
        assert "60 second command timeout" in manager._current_policy.prompt

    def test_configure_for_task_no_execution_config(self):
        """Test configuring PolicyManager with task that has no execution_config."""
        manager = PolicyManager("test_domain")

        # Mock task without execution config
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = None

        # Configure for task
        manager.configure_for_task("session_123", mock_task)

        # Verify default timeout was used
        assert manager._timeout_seconds == 60
        assert manager._current_policy is not None

    def test_get_policy_after_configure(self):
        """Test getting policy after configure_for_task."""
        manager = PolicyManager("test_domain")

        # Mock task with custom timeout
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 90}

        # Configure for task
        manager.configure_for_task("session_123", mock_task)

        # Get policy
        policy = manager.get_policy()

        # Verify policy contains the configured timeout
        assert isinstance(policy, PolicyDocument)
        assert "90 second command timeout" in policy.prompt
        assert policy == manager._current_policy

    def test_get_policy_before_configure(self):
        """Test getting policy before configure_for_task (should use defaults)."""
        manager = PolicyManager("test_domain")

        # Get policy without configuring first
        policy = manager.get_policy()

        # Should return default policy with default timeout
        assert isinstance(policy, PolicyDocument)
        assert "60 second command timeout" in policy.prompt

    def test_policy_document_structure(self):
        """Test that generated policy document has expected structure."""
        manager = PolicyManager("test_domain")

        # Mock task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 180}

        # Configure and get policy
        manager.configure_for_task("session_123", mock_task)
        policy = manager.get_policy()

        # Verify policy document structure and content
        assert hasattr(policy, "prompt")
        assert isinstance(policy.prompt, str)
        assert len(policy.prompt) > 0

        # Check for key content elements
        assert "SABER framework" in policy.prompt
        assert "FLAG CAPTURE MISSION" in policy.prompt
        assert "180 second command timeout" in policy.prompt
        assert "BEGIN AUTONOMOUS FLAG CAPTURE MISSION NOW" in policy.prompt

    @pytest.mark.asyncio
    async def test_validate_action(self):
        """Test action validation (stub implementation)."""
        manager = PolicyManager("test_domain")

        # Test validation
        is_valid = await manager.validate_action("test_action")

        # Stub implementation should always return True
        assert is_valid is True
