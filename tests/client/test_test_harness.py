"""
Tests for TestHarness functionality.

Tests the main orchestrator that coordinates server, agent, and prompt building.
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, patch, MagicMock
from typing import Any, Dict, List
from pathlib import Path

from saber.client.test_harness import TestHarness, TestHarnessConfig
from saber.client.server_client import ServerClient
from saber.client.prompt_builder import PromptBuilder
from saber.client.agent_wrapper import AgentWrapper
from saber.api_models import StepResponse, TaskInfo, PolicyInfo, EpisodeInfo


@pytest.fixture
def test_agent():
    """Mock agent for testing."""
    agent = AsyncMock()
    agent.process_prompt = AsyncMock(return_value="agent response")
    agent.reset = AsyncMock()
    return agent


@pytest.fixture
def mock_server_client():
    """Mock ServerClient for testing."""
    client = AsyncMock(spec=ServerClient)
    client.session_id = "test-session"
    client.create_session = AsyncMock(return_value="test-session")
    client.start_episode = AsyncMock(return_value=EpisodeInfo(
        episode_id="episode-123",
        task_id="task-456",
        message="Episode started"
    ))
    client.execute_step = AsyncMock(return_value=StepResponse(
        success=True,
        output="command output",
        done=False,
        error=None,
        info={}
    ))
    client.get_current_task = AsyncMock(return_value=TaskInfo(
        task_id="task-456",
        title="Test Task",
        description="Test description",
        current_subtask="subtask-1"
    ))
    client.get_policy = AsyncMock(return_value=PolicyInfo(
        domain="test",
        available_commands=["ls", "cat"],
        guidelines="Be careful",
        constraints=["No rm commands"]
    ))
    client.close_session = AsyncMock()
    return client


@pytest.fixture
def mock_prompt_builder():
    """Mock PromptBuilder for testing."""
    builder = AsyncMock(spec=PromptBuilder)
    builder.build_initial_prompt = AsyncMock(return_value="Initial prompt")
    builder.build_step_prompt = AsyncMock(return_value="Step prompt")
    return builder


@pytest.fixture
def mock_agent_wrapper():
    """Mock AgentWrapper for testing."""
    wrapper = AsyncMock(spec=AgentWrapper)
    wrapper.invoke = AsyncMock(return_value="agent response")
    wrapper.reset = AsyncMock()
    return wrapper


@pytest.fixture
def test_config():
    """Test configuration for TestHarness."""
    return TestHarnessConfig(
        server_url="http://test:8000",
        max_steps=5,
        request_timeout=30.0,
        log_level="INFO"
    )


class TestTestHarnessConfig:
    """Test cases for TestHarnessConfig."""

    def test_default_values(self):
        """Test TestHarnessConfig with default values."""
        config = TestHarnessConfig(
            server_url="http://localhost:8000"
        )

        assert config.server_url == "http://localhost:8000"
        assert config.max_steps == 100
        assert config.request_timeout == 30.0
        assert config.log_level == "INFO"
        assert config.log_file is None

    def test_custom_values(self, test_config):
        """Test TestHarnessConfig with custom values."""
        assert test_config.server_url == "http://test:8000"
        assert test_config.max_steps == 5
        assert test_config.request_timeout == 30.0
        assert test_config.log_level == "INFO"


class TestTestHarness:
    """Test cases for TestHarness."""

    def test_init(self, test_config, test_agent):
        """Test TestHarness initialization."""
        harness = TestHarness(test_config)

        assert harness.config == test_config
        assert harness.server_client is None  # Not initialized yet
        assert harness.agent is None  # Not initialized yet

    @pytest.mark.asyncio
    async def test_basic_harness_functionality(self, test_config, test_agent):
        """Test basic harness setup and teardown."""
        harness = TestHarness(test_config)

        # Mock the ServerClient to avoid real network calls
        with patch('saber.client.test_harness.ServerClient') as MockServerClient:
            mock_server = AsyncMock()
            MockServerClient.return_value = mock_server

            # Test initialization
            await harness.initialize(test_agent)
            assert harness.agent == test_agent
            assert harness.server_client is not None

            # Test shutdown
            await harness.shutdown()

    @pytest.mark.asyncio
    async def test_context_manager(self, test_config):
        """Test TestHarness as async context manager."""
        async with TestHarness(test_config) as harness:
            assert harness is not None
            assert hasattr(harness, 'config')

    def test_basic_methods_exist(self, test_config):
        """Test that required methods exist."""
        harness = TestHarness(test_config)

        # Check that key methods exist
        assert hasattr(harness, 'initialize')
        assert hasattr(harness, 'run_test')
        assert hasattr(harness, 'shutdown')
        assert hasattr(harness, '__aenter__')
        assert hasattr(harness, '__aexit__')


class TestIntegrationScenarios:
    """Integration test scenarios for TestHarness."""

    @pytest.mark.asyncio
    async def test_basic_integration_workflow(self, test_agent):
        """Test basic integration workflow."""
        config = TestHarnessConfig(
            server_url="http://test:8000",
            max_steps=3,
            request_timeout=30.0
        )

        harness = TestHarness(config)

        # Mock the ServerClient to avoid real network calls
        with patch('saber.client.test_harness.ServerClient') as MockServerClient:
            mock_server = AsyncMock()
            MockServerClient.return_value = mock_server

            # Test that we can create and initialize
            await harness.initialize(test_agent)
            assert harness.agent == test_agent

            # Cleanup
            await harness.shutdown()

    @pytest.mark.asyncio
    async def test_context_manager_integration(self, test_agent):
        """Test context manager integration."""
        config = TestHarnessConfig(
            server_url="http://test:8000",
            max_steps=2,
            request_timeout=30.0
        )

        # Mock the ServerClient to avoid real network calls
        with patch('saber.client.test_harness.ServerClient') as MockServerClient:
            mock_server = AsyncMock()
            MockServerClient.return_value = mock_server

            # Test context manager usage
            async with TestHarness(config) as harness:
                await harness.initialize(test_agent)
                assert harness.agent == test_agent
