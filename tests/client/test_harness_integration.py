#!/usr/bin/env python3
"""
Integration test for the new SABER harness implementation.

Creates a simple test agent and validates the harness can run it.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from saber.client import SABERHarness, SABERHarnessConfig


class SimpleTestAgent:
    """Simple test agent for integration testing."""

    def __init__(self, mcp_client):
        self.mcp_client = mcp_client
        self.call_count = 0

    async def run(self, initial_prompt: str, shutdown_check=None) -> dict:
        """Run the agent with MCP tool calls."""
        # Simulate some tool calls
        tools = await self.mcp_client.list_tools()

        # Make a couple of tool calls
        for i in range(3):
            if shutdown_check and shutdown_check():
                break

            # Simulate tool call
            await self.mcp_client.call_tool("test_tool", {"arg": f"value_{i}"})
            self.call_count += 1

        return {
            "success": True,
            "flag": "flag{test_success}",
            "message": f"Agent completed with {self.call_count} tool calls"
        }


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


def create_mock_mcp_client():
    """Create a mock MCP client for testing."""
    mock_mcp = AsyncMock()

    # Mock tool listing
    mock_mcp.list_tools.return_value = [
        {"name": "test_tool", "description": "A test tool"}
    ]

    # Mock tool calls
    mock_mcp.call_tool.return_value = {
        "content": [{"type": "text", "text": "Tool executed successfully"}]
    }

    # Mock connection
    mock_mcp.connect.return_value = None
    mock_mcp.disconnect.return_value = None

    # Mock termination properties
    mock_mcp.episode_terminated = False
    mock_mcp.termination_reason = None

    return mock_mcp


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

    # Mock the REST and MCP clients
    harness.rest_client = create_mock_rest_client()

    # Mock MCP client creation
    original_create_mcp = harness._create_mcp_client
    async def mock_create_mcp(task_id):
        mock_mcp = create_mock_mcp_client()
        mock_mcp._step_count = 0
        return mock_mcp
    harness._create_mcp_client = mock_create_mcp

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
    harness._create_mcp_client = lambda task_id: create_mock_mcp_client()

    # Mock the MCP client creation
    async def mock_create_mcp(task_id):
        mock_mcp = create_mock_mcp_client()
        mock_mcp._step_count = 0
        return mock_mcp
    harness._create_mcp_client = mock_create_mcp

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
