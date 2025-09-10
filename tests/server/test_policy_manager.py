"""
Unit tests for PolicyManager.
"""

import pytest
from unittest.mock import MagicMock

from saber.server.policy.policy_manager import PolicyManager, PolicyDocument


class TestPolicyManager:
    """Test cases for PolicyManager functionality."""

    def test_policy_manager_init(self):
        """Test PolicyManager initialization."""
        manager = PolicyManager("test_domain")
        assert manager.domain_name == "test_domain"
        assert manager._episode_policies == {}  # Episode storage

    def test_set_episode_policy_with_prompt(self):
        """Test setting episode policy with pre-generated prompt."""
        manager = PolicyManager("test_domain")

        # Test setting policy with a valid prompt
        episode_id = "episode_123"
        session_id = "session_456"
        prompt = "You are a test agent. Complete the security task using available tools."

        manager.set_episode_policy(episode_id, session_id, prompt)

        # Verify policy was stored
        policy = manager.get_policy(episode_id)
        assert policy.prompt == prompt

    def test_set_episode_policy_empty_prompt_fails(self):
        """Test setting episode policy with empty prompt fails."""
        manager = PolicyManager("test_domain")

        # Test with empty prompt
        with pytest.raises(ValueError, match="Non-empty pre_generated_prompt required"):
            manager.set_episode_policy("episode_123", "session_456", "")

        # Test with whitespace-only prompt
        with pytest.raises(ValueError, match="Non-empty pre_generated_prompt required"):
            manager.set_episode_policy("episode_123", "session_456", "   \n  \t  ")

    def test_get_policy_after_setting(self):
        """Test getting policy after setting it."""
        manager = PolicyManager("test_domain")

        episode_id = "episode_123"
        session_id = "session_456"
        prompt = "Test prompt for episode 123"

        # Set policy
        manager.set_episode_policy(episode_id, session_id, prompt)

        # Get policy
        policy = manager.get_policy(episode_id)
        assert isinstance(policy, PolicyDocument)
        assert policy.prompt == prompt

    def test_get_policy_unconfigured_episode(self):
        """Test getting policy for unconfigured episode fails."""
        manager = PolicyManager("test_domain")

        with pytest.raises(ValueError, match="No policy configured for episode"):
            manager.get_policy("nonexistent_episode")

    def test_get_policy_requires_episode_id(self):
        """Test that get_policy requires episode_id parameter."""
        manager = PolicyManager("test_domain")

        with pytest.raises(ValueError, match="episode_id is required"):
            manager.get_policy(None)

    def test_policy_document_structure(self):
        """Test PolicyDocument structure and methods."""
        manager = PolicyManager("test_domain")

        episode_id = "episode_123"
        session_id = "session_456"
        prompt = "Test prompt with specific content"

        manager.set_episode_policy(episode_id, session_id, prompt)

        # Get policy and verify structure
        policy = manager.get_policy(episode_id)
        assert hasattr(policy, "prompt")
        assert hasattr(policy, "to_dict")

        # Test to_dict method
        policy_dict = policy.to_dict()
        assert isinstance(policy_dict, dict)
        assert "prompt" in policy_dict
        assert policy_dict["prompt"] == prompt

    def test_cleanup_episode_policy(self):
        """Test cleaning up episode policy."""
        manager = PolicyManager("test_domain")

        episode_id = "episode_123"
        session_id = "session_456"
        prompt = "Test prompt"

        # Set policy
        manager.set_episode_policy(episode_id, session_id, prompt)
        assert episode_id in manager._episode_policies

        # Clean up policy
        manager.cleanup_episode_policy(episode_id)
        assert episode_id not in manager._episode_policies

        # Trying to get policy should fail now
        with pytest.raises(ValueError, match="No policy configured for episode"):
            manager.get_policy(episode_id)

    def test_cleanup_nonexistent_episode(self):
        """Test cleaning up policy for non-existent episode doesn't fail."""
        manager = PolicyManager("test_domain")

        # This should not raise an exception
        manager.cleanup_episode_policy("nonexistent_episode")

    def test_episode_isolation(self):
        """Test that episodes have isolated policies."""
        manager = PolicyManager("test_domain")

        # Set policies for two different episodes
        episode1_id = "episode_1"
        episode2_id = "episode_2"
        session_id = "session_456"
        prompt1 = "Prompt for episode 1"
        prompt2 = "Prompt for episode 2"

        manager.set_episode_policy(episode1_id, session_id, prompt1)
        manager.set_episode_policy(episode2_id, session_id, prompt2)

        # Verify they are isolated
        policy1 = manager.get_policy(episode1_id)
        policy2 = manager.get_policy(episode2_id)

        assert policy1.prompt == prompt1
        assert policy2.prompt == prompt2
        assert policy1.prompt != policy2.prompt

        # Clean up one episode
        manager.cleanup_episode_policy(episode1_id)

        # Episode 2 should still work
        policy2_again = manager.get_policy(episode2_id)
        assert policy2_again.prompt == prompt2

        # Episode 1 should no longer work
        with pytest.raises(ValueError, match="No policy configured for episode"):
            manager.get_policy(episode1_id)

    @pytest.mark.asyncio
    async def test_validate_action(self):
        """Test action validation (stub implementation)."""
        manager = PolicyManager("test_domain")

        # Current implementation always returns True (stub)
        result = await manager.validate_action("test_action")
        assert result is True

        result = await manager.validate_action("any_action")
        assert result is True


if __name__ == "__main__":
    pytest.main([__file__])
