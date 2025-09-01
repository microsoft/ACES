#!/usr/bin/env python3
"""
Integration test for the new SABER harness implementation.

Creates a simple test agent and validates the harness can run it.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from saber.client import SABERHarness, SABERHarnessConfig


class SimpleTestAgent:
    """Simple test agent placeholder for container execution tests."""

    def __init__(self, *args, **kwargs):
        self.call_count = 0

    async def run(self, initial_prompt: str, shutdown_check=None) -> dict:
        # In container mode, the agent logic executes inside the container.
        # This placeholder isn't used by the mocked container executor.
        self.call_count = 3
        return {"success": True, "flag": "flag{test_success}", "message": "ok"}


def create_mock_rest_client():
    """Create a mock REST client for testing."""
    mock_rest = AsyncMock()

    # Mock session creation
    mock_rest.create_session.return_value = "test_session_123"

    # Mock task listing
    mock_rest.list_tasks.return_value = [
        {"task_id": "test_task_1", "title": "Test Task 1"},
        {"task_id": "test_task_2", "title": "Test Task 2"},
    ]

    # Mock episode start
    mock_rest.start_episode.return_value = "episode_456"

    # Mock policy retrieval
    mock_rest.get_policy_info.return_value = {
        "prompt": "You are a security testing agent. Complete the task using available tools."
    }

    # Mock session termination
    mock_rest.terminate_session.return_value = True

    return mock_rest


class MockContainerExecutor:
    """Mock container executor that returns successful episode results."""

    async def initialize(self):
        return None

    async def execute_episodes(self, episodes, agent):
        from saber.client.harness_models import EpisodeResult

        results = []
        for task_id, attempt in episodes:
            results.append(
                EpisodeResult(
                    task_id=task_id,
                    episode_id="episode_456",
                    attempt=attempt,
                    success=True,
                    termination_reason="completed",
                    flag="flag{test_success}",
                    iterations=3,
                )
            )
        return results

    async def cleanup(self):
        return None


@pytest.mark.asyncio
async def test_harness_integration():
    """Test complete harness functionality with mocked dependencies."""
    print("🧪 Starting SABER harness integration test...")

    # Create configuration
    config = SABERHarnessConfig(
        server_url="http://localhost:8000",
        mcp_url="http://localhost:8001",
        task_ids=["test_task_1"],  # Test specific task selection
        parallelism=1,
        max_steps_client_safety=10
    )

    # Create harness
    harness = SABERHarness(config)

    # Create test agent
    agent = SimpleTestAgent

    # Initialize harness
    await harness.initialize(agent)
    print("✅ Harness initialized")

    # Mock the REST client
    harness.rest_client = create_mock_rest_client()
    # Inject a mock container executor that returns successful results
    harness.container_executor = MockContainerExecutor()

    # Run harness
    print("🚀 Running harness...")
    results = await harness.run()

    # Validate results
    print("📊 Validating results...")
    assert results.success, "Harness should report success"
    assert results.total_episodes == 1, f"Expected 1 episode, got {results.total_episodes}"
    assert results.successful_episodes == 1, f"Expected 1 successful episode, got {results.successful_episodes}"
    assert len(results.episode_results) == 1, f"Expected 1 episode result, got {len(results.episode_results)}"

    episode_result = results.episode_results[0]
    assert episode_result.task_id == "test_task_1", f"Expected task_id test_task_1, got {episode_result.task_id}"
    assert episode_result.success, "Episode should be successful"
    assert episode_result.flag == "flag{test_success}", f"Expected flag flag{{test_success}}, got {episode_result.flag}"

    print("✅ All assertions passed!")
    print(f"📋 Results: {results.successful_episodes}/{results.total_episodes} episodes successful")
    print(f"🎯 Episode details: {episode_result.task_id} - {episode_result.termination_reason}")


@pytest.mark.asyncio
async def test_harness_with_multiple_tasks():
    """Test harness with multiple tasks (parallelism=1)."""
    print("\n🧪 Testing harness with multiple tasks...")

    config = SABERHarnessConfig(
        task_ids=None,  # Test auto-fetch all tasks
        parallelism=1
    )

    harness = SABERHarness(config)
    await harness.initialize(SimpleTestAgent)

    # Mock clients
    harness.rest_client = create_mock_rest_client()
    harness.container_executor = MockContainerExecutor()

    results = await harness.run()

    # Should run both tasks from the mock
    assert results.total_episodes == 2, f"Expected 2 episodes, got {results.total_episodes}"
    assert results.successful_episodes == 2, f"Expected 2 successful episodes, got {results.successful_episodes}"

    task_ids = {r.task_id for r in results.episode_results}
    assert task_ids == {"test_task_1", "test_task_2"}, f"Expected both tasks, got {task_ids}"

    print("✅ Multiple tasks test passed!")


if __name__ == "__main__":
    async def main():
        await test_harness_integration()
        await test_harness_with_multiple_tasks()
        print("\n🎉 All integration tests passed!")

    asyncio.run(main())
