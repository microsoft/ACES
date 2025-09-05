"""
Unit tests for PolicyManager.
Tests policy document generation and episode-specific configuration.
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
        assert manager._default_timeout_seconds == 60  # Default timeout
        assert manager._episode_policies == {}  # Episode storage

    def test_configure_for_episode_with_timeout(self):
        """Test configuring PolicyManager with task that has timeout."""
        manager = PolicyManager("test_domain")

        # Mock task with execution config containing timeout
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 120, "other_config": "value"}

        # Configure for episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)

        # Verify policy was created for episode
        assert "episode_123" in manager._episode_policies
        episode_policy = manager._episode_policies["episode_123"]
        assert isinstance(episode_policy, PolicyDocument)

        # Verify the timeout appears in the policy prompt
        assert "120 second command timeout" in episode_policy.prompt

    def test_configure_for_episode_without_timeout(self):
        """Test configuring PolicyManager with task that has no timeout."""
        manager = PolicyManager("test_domain")

        # Mock task without execution config
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {}

        # Configure for episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)

        # Verify policy was created with default timeout
        assert "episode_123" in manager._episode_policies
        episode_policy = manager._episode_policies["episode_123"]
        assert isinstance(episode_policy, PolicyDocument)

        # Verify the default timeout appears in the policy prompt
        assert "60 second command timeout" in episode_policy.prompt

    def test_configure_for_episode_no_execution_config(self):
        """Test configuring PolicyManager with task that has no execution_config."""
        manager = PolicyManager("test_domain")

        # Mock task without execution config
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = None

        # Configure for episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)

        # Verify policy was created with default timeout
        assert "episode_123" in manager._episode_policies
        episode_policy = manager._episode_policies["episode_123"]
        assert isinstance(episode_policy, PolicyDocument)

    def test_get_policy_after_configure(self):
        """Test getting policy after configure_for_episode."""
        manager = PolicyManager("test_domain")

        # Mock task with custom timeout
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 90}

        # Configure for episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)

        # Get policy for episode
        policy = manager.get_policy("episode_123")

        # Verify policy contains the configured timeout
        assert isinstance(policy, PolicyDocument)
        assert "90 second command timeout" in policy.prompt

    def test_get_policy_unconfigured_episode(self):
        """Test getting policy for unconfigured episode should fail fast."""
        manager = PolicyManager("test_domain")

        # Attempt to get policy for episode that hasn't been configured
        with pytest.raises(ValueError, match="No policy configured for episode episode_unknown"):
            manager.get_policy("episode_unknown")

    def test_get_policy_requires_episode_id(self):
        """Test getting policy without episode_id should fail fast."""
        manager = PolicyManager("test_domain")

        # Attempt to get policy without episode_id
        with pytest.raises(ValueError, match="episode_id is required for policy retrieval"):
            manager.get_policy(None)

    def test_policy_document_structure(self):
        """Test that generated policy document has expected structure."""
        manager = PolicyManager("test_domain")

        # Mock task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 180}

        # Configure and get policy for episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)
        policy = manager.get_policy("episode_123")

        # Verify policy document structure and content
        assert hasattr(policy, "prompt")
        assert isinstance(policy.prompt, str)
        assert len(policy.prompt) > 0

        # Check for key content elements
        assert "SABER framework" in policy.prompt
        assert "FLAG CAPTURE MISSION" in policy.prompt
        assert "180 second command timeout" in policy.prompt
        assert "BEGIN AUTONOMOUS FLAG CAPTURE MISSION NOW" in policy.prompt

    def test_cleanup_episode_policy(self):
        """Test cleaning up policy for completed episode."""
        manager = PolicyManager("test_domain")

        # Mock task
        mock_task = MagicMock()
        mock_task.task_id = "test_task"
        mock_task.execution_config = {"timeout": 90}

        # Configure episode
        manager.configure_for_episode("episode_123", "session_456", mock_task)
        assert "episode_123" in manager._episode_policies

        # Clean up episode policy
        manager.cleanup_episode_policy("episode_123")
        assert "episode_123" not in manager._episode_policies

    def test_episode_isolation(self):
        """Test that episodes have isolated policies."""
        manager = PolicyManager("test_domain")

        # Configure two episodes with different timeouts
        mock_task1 = MagicMock()
        mock_task1.task_id = "task1"
        mock_task1.execution_config = {"timeout": 60}

        mock_task2 = MagicMock()
        mock_task2.task_id = "task2"
        mock_task2.execution_config = {"timeout": 120}

        manager.configure_for_episode("episode_1", "session_456", mock_task1)
        manager.configure_for_episode("episode_2", "session_456", mock_task2)

        # Verify each episode has its own policy with correct timeout
        policy1 = manager.get_policy("episode_1")
        policy2 = manager.get_policy("episode_2")

        assert "60 second command timeout" in policy1.prompt
        assert "120 second command timeout" in policy2.prompt
        assert policy1.prompt != policy2.prompt

    def test_multi_episode_orchestration_same_session(self):
        """Test PolicyManager can orchestrate multiple episodes within the same session."""
        manager = PolicyManager("test_domain")
        session_id = "session_multi_episode"

        # Configure multiple episodes for the same session with different tasks
        episodes_config = [
            ("episode_web_scan", {"timeout": 30}, "web_scanner"),
            ("episode_vuln_exploit", {"timeout": 180}, "exploit_runner"),
            ("episode_data_extract", {"timeout": 90}, "data_extractor"),
        ]

        # Configure all episodes
        for episode_id, exec_config, task_name in episodes_config:
            mock_task = MagicMock()
            mock_task.task_id = task_name
            mock_task.execution_config = exec_config

            manager.configure_for_episode(episode_id, session_id, mock_task)

        # Verify all episodes are independently configured
        assert len(manager._episode_policies) == 3
        assert "episode_web_scan" in manager._episode_policies
        assert "episode_vuln_exploit" in manager._episode_policies
        assert "episode_data_extract" in manager._episode_policies

        # Verify each episode has its unique policy with correct timeout
        web_policy = manager.get_policy("episode_web_scan")
        exploit_policy = manager.get_policy("episode_vuln_exploit")
        extract_policy = manager.get_policy("episode_data_extract")

        assert "30 second command timeout" in web_policy.prompt
        assert "180 second command timeout" in exploit_policy.prompt
        assert "90 second command timeout" in extract_policy.prompt

        # Verify policies are truly independent (different content)
        assert web_policy.prompt != exploit_policy.prompt != extract_policy.prompt

        # Test episode lifecycle - cleanup one episode shouldn't affect others
        manager.cleanup_episode_policy("episode_web_scan")
        assert "episode_web_scan" not in manager._episode_policies
        assert "episode_vuln_exploit" in manager._episode_policies
        assert "episode_data_extract" in manager._episode_policies

        # Remaining episodes should still be accessible
        remaining_exploit_policy = manager.get_policy("episode_vuln_exploit")
        remaining_extract_policy = manager.get_policy("episode_data_extract")
        assert "180 second command timeout" in remaining_exploit_policy.prompt
        assert "90 second command timeout" in remaining_extract_policy.prompt

    def test_concurrent_multi_session_multi_episode_orchestration(self):
        """Test PolicyManager can handle multiple episodes across multiple sessions concurrently."""
        manager = PolicyManager("test_domain")

        # Simulate concurrent sessions each running multiple episodes
        session_episodes = {
            "session_pentest_1": [
                ("episode_recon", {"timeout": 60}, "reconnaissance"),
                ("episode_attack", {"timeout": 300}, "attack_phase"),
            ],
            "session_pentest_2": [
                ("episode_scan", {"timeout": 45}, "port_scanner"),
                ("episode_enum", {"timeout": 120}, "service_enum"),
                ("episode_privesc", {"timeout": 240}, "privilege_escalation"),
            ],
            "session_forensics": [
                ("episode_collect", {"timeout": 180}, "evidence_collection"),
            ]
        }

        # Configure all episodes across all sessions
        total_episodes = 0
        for session_id, episodes in session_episodes.items():
            for episode_id, exec_config, task_name in episodes:
                mock_task = MagicMock()
                mock_task.task_id = task_name
                mock_task.execution_config = exec_config

                manager.configure_for_episode(episode_id, session_id, mock_task)
                total_episodes += 1

        # Verify all episodes are registered
        assert len(manager._episode_policies) == total_episodes == 6

        # Verify cross-session episode isolation - each episode should be independently accessible
        test_cases = [
            ("episode_recon", "60 second command timeout"),
            ("episode_attack", "300 second command timeout"),
            ("episode_scan", "45 second command timeout"),
            ("episode_enum", "120 second command timeout"),
            ("episode_privesc", "240 second command timeout"),
            ("episode_collect", "180 second command timeout"),
        ]

        for episode_id, expected_timeout in test_cases:
            policy = manager.get_policy(episode_id)
            assert expected_timeout in policy.prompt

        # Test selective cleanup - cleanup one session's episodes
        pentest1_episodes = ["episode_recon", "episode_attack"]
        for episode_id in pentest1_episodes:
            manager.cleanup_episode_policy(episode_id)

        # Verify only session_pentest_1 episodes were cleaned up
        assert len(manager._episode_policies) == 4  # 6 - 2 = 4

        # session_pentest_1 episodes should be gone
        for episode_id in pentest1_episodes:
            with pytest.raises(ValueError, match=f"No policy configured for episode {episode_id}"):
                manager.get_policy(episode_id)

        # Other sessions' episodes should remain unaffected
        remaining_episodes = ["episode_scan", "episode_enum", "episode_privesc", "episode_collect"]
        for episode_id in remaining_episodes:
            policy = manager.get_policy(episode_id)
            assert "SABER framework" in policy.prompt  # Verify policy is still valid

        # Verify episode isolation - can still configure new episodes even after cleanup
        new_task = MagicMock()
        new_task.task_id = "new_task"
        new_task.execution_config = {"timeout": 75}

        manager.configure_for_episode("episode_new", "session_new", new_task)
        new_policy = manager.get_policy("episode_new")
        assert "75 second command timeout" in new_policy.prompt
        assert len(manager._episode_policies) == 5  # 4 + 1 = 5

    @pytest.mark.asyncio
    async def test_validate_action(self):
        """Test action validation (stub implementation)."""
        manager = PolicyManager("test_domain")

        # Test validation
        is_valid = await manager.validate_action("test_action")

        # Stub implementation should always return True
        assert is_valid is True
