#!/usr/bin/env python3
"""
Tests for SABER Harness Implementation

Basic unit tests to validate the new harness implementation according to the plan.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from saber.client.agent_wrapper import AgentWrapper
from saber.client.harness_models import SABERHarnessConfig, EpisodeResult
from saber.client.saber_harness import SABERHarness


class TestAgentWrapper:
    """Test AgentWrapper signature detection and normalization."""

    def test_class_agent_with_run_method(self):
        """Test agent class with run method."""
        class TestAgent:
            def __init__(self, mcp_client):
                self.mcp_client = mcp_client

            def run(self, prompt: str) -> str:
                return f"Response to: {prompt}"

        mock_mcp = MagicMock()
        wrapper = AgentWrapper(TestAgent, mock_mcp)

        assert wrapper.agent_instance is not None
        assert wrapper.agent_callable is not None

    def test_callable_agent(self):
        """Test directly callable agent."""
        def test_agent(prompt: str) -> str:
            return f"Response to: {prompt}"

        mock_mcp = MagicMock()
        wrapper = AgentWrapper(test_agent, mock_mcp)

        # For functions, agent_callable should be the function itself
        assert wrapper.agent_callable == test_agent

    def test_class_agent_with_call_method(self):
        """Test agent class with __call__ method."""
        class TestAgent:
            def __init__(self, mcp_client):
                self.mcp_client = mcp_client

            def __call__(self, prompt: str) -> str:
                return f"Response to: {prompt}"

        mock_mcp = MagicMock()
        wrapper = AgentWrapper(TestAgent, mock_mcp)

        assert wrapper.agent_instance is not None
        assert wrapper.agent_callable is not None

    def test_shutdown_check(self):
        """Test shutdown check functionality."""
        def test_agent(prompt: str) -> str:
            return f"Response to: {prompt}"

        mock_mcp = MagicMock()
        wrapper = AgentWrapper(test_agent, mock_mcp)

        assert not wrapper.shutdown_check()
        wrapper.set_shutdown()
        assert wrapper.shutdown_check()

    @pytest.mark.asyncio
    async def test_run_with_different_signatures(self):
        """Test running agents with different parameter signatures."""

        # Agent with shutdown_check
        def agent_with_shutdown(prompt: str, shutdown_check) -> str:
            return f"Got prompt: {prompt}, shutdown: {shutdown_check()}"

        mock_mcp = MagicMock()
        wrapper = AgentWrapper(agent_with_shutdown, mock_mcp)
        result = await wrapper.run("test prompt")
        assert "Got prompt: test prompt" in result
        assert "shutdown: False" in result

        # Agent without shutdown_check
        def simple_agent(prompt: str) -> str:
            return f"Simple response to: {prompt}"

        wrapper2 = AgentWrapper(simple_agent, mock_mcp)
        result2 = await wrapper2.run("test prompt")
        assert result2 == "Simple response to: test prompt"


class TestSABERHarness:
    """Test SABERHarness core functionality."""

    def test_config_defaults(self):
        """Test default configuration values."""
        config = SABERHarnessConfig()

        assert config.server_url == "http://localhost:8000"
        assert config.mcp_url == "http://localhost:8001"
        assert config.client_id == "saber-client"
        assert config.parallelism == 1
        assert config.max_steps_client_safety == 100
        assert config.task_ids is None

    def test_config_custom_values(self):
        """Test custom configuration values."""
        config = SABERHarnessConfig(
            server_url="http://custom:9000",
            task_ids=["task1", "task2"],
            parallelism=4,
            max_steps_client_safety=200
        )

        assert config.server_url == "http://custom:9000"
        assert config.task_ids == ["task1", "task2"]
        assert config.parallelism == 4
        assert config.max_steps_client_safety == 200

    @pytest.mark.asyncio
    async def test_harness_initialization(self):
        """Test harness initialization."""
        def test_agent(prompt: str) -> str:
            return "test response"

        config = SABERHarnessConfig()
        harness = SABERHarness(config)

        await harness.initialize(test_agent)

        assert harness.agent == test_agent
        assert harness.rest_client is not None
        assert harness.config == config

    def test_episode_result_creation(self):
        """Test EpisodeResult data model."""
        result = EpisodeResult(
            task_id="test_task",
            episode_id="ep_123",
            attempt=1,
            success=True,
            termination_reason="completed",
            flag="flag{test}",
            iterations=5
        )

        assert result.task_id == "test_task"
        assert result.episode_id == "ep_123"
        assert result.success is True
        assert result.termination_reason == "completed"
        assert result.flag == "flag{test}"
        assert result.iterations == 5


class TestTaskResolution:
    """Test task resolution logic (R2/R2b)."""

    @pytest.mark.asyncio
    async def test_supplied_tasks(self):
        """Test R2: supplied tasks are used exactly."""
        config = SABERHarnessConfig(task_ids=["task1", "task2"])
        harness = SABERHarness(config)

        # Mock REST client
        mock_rest = AsyncMock()
        mock_rest.list_tasks.return_value = [
            {"task_id": "task1", "title": "Task 1"},
            {"task_id": "task2", "title": "Task 2"},
            {"task_id": "task3", "title": "Task 3"},  # Should be filtered out
        ]
        harness.rest_client = mock_rest

        tasks = await harness._resolve_tasks()

        assert len(tasks) == 2
        assert {t["task_id"] for t in tasks} == {"task1", "task2"}

    @pytest.mark.asyncio
    async def test_auto_fetch_all_tasks(self):
        """Test R2b: auto-fetch all available tasks."""
        config = SABERHarnessConfig(task_ids=None)  # No tasks specified
        harness = SABERHarness(config)

        # Mock REST client
        mock_rest = AsyncMock()
        mock_rest.list_tasks.return_value = [
            {"task_id": "task1", "title": "Task 1"},
            {"task_id": "task2", "title": "Task 2"},
            {"task_id": "task3", "title": "Task 3"},
        ]
        harness.rest_client = mock_rest

        tasks = await harness._resolve_tasks()

        assert len(tasks) == 3
        assert {t["task_id"] for t in tasks} == {"task1", "task2", "task3"}

    @pytest.mark.asyncio
    async def test_missing_supplied_tasks(self):
        """Test error handling for missing supplied tasks."""
        config = SABERHarnessConfig(task_ids=["nonexistent"])
        harness = SABERHarness(config)

        # Mock REST client
        mock_rest = AsyncMock()
        mock_rest.list_tasks.return_value = [
            {"task_id": "task1", "title": "Task 1"},
        ]
        harness.rest_client = mock_rest

        with pytest.raises(ValueError, match="No matching tasks found"):
            await harness._resolve_tasks()


if __name__ == "__main__":
    pytest.main([__file__])
