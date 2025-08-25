"""
Tests for TestHarness functionality.

Tests the main orchestrator that coordinates server, agent, and prompt building.
"""

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from saber.api_models import EpisodeInfo, PolicyInfo, StepResponse, TaskInfo
from saber.client.agent_wrapper import AgentWrapper
from saber.client.prompt_builder import PromptBuilder
from saber.client.server_client import ServerClient
from saber.client.saber_harness import TestHarness, TestHarnessConfig


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
    client.start_episode = AsyncMock(
        return_value=EpisodeInfo(episode_id="episode-123", task_id="task-456", message="Episode started")
    )
    client.execute_step = AsyncMock(
        return_value=StepResponse(success=True, output="command output", done=False, error=None, info={})
    )
    client.get_current_task = AsyncMock(
        return_value=TaskInfo(
            task_id="task-456", title="Test Task", description="Test description", current_subtask="subtask-1"
        )
    )
    client.get_policy = AsyncMock(
        return_value=PolicyInfo(
            domain="test", available_commands=["ls", "cat"], guidelines="Be careful", constraints=["No rm commands"]
        )
    )
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
    return TestHarnessConfig(server_url="http://test:8000", max_steps=5, request_timeout=30.0, log_level="INFO")


class TestTestHarnessConfig:
    """Test cases for TestHarnessConfig."""

    def test_default_values(self):
        """Test TestHarnessConfig with default values."""
        config = TestHarnessConfig(server_url="http://localhost:8000")

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
        with patch("saber.client.test_harness.ServerClient") as MockServerClient:
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
            assert hasattr(harness, "config")

    def test_basic_methods_exist(self, test_config):
        """Test that required methods exist."""
        harness = TestHarness(test_config)

        # Check that key methods exist
        assert hasattr(harness, "initialize")
        assert hasattr(harness, "run_test")
        assert hasattr(harness, "shutdown")
        assert hasattr(harness, "__aenter__")
        assert hasattr(harness, "__aexit__")


class TestIntegrationScenarios:
    """Integration test scenarios for TestHarness."""

    @pytest.mark.asyncio
    async def test_basic_integration_workflow(self, test_agent):
        """Test basic integration workflow."""
        config = TestHarnessConfig(server_url="http://test:8000", max_steps=3, request_timeout=30.0)

        harness = TestHarness(config)

        # Mock the ServerClient to avoid real network calls
        with patch("saber.client.test_harness.ServerClient") as MockServerClient:
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
        config = TestHarnessConfig(server_url="http://test:8000", max_steps=2, request_timeout=30.0)

        # Mock the ServerClient to avoid real network calls
        with patch("saber.client.test_harness.ServerClient") as MockServerClient:
            mock_server = AsyncMock()
            MockServerClient.return_value = mock_server

            # Test context manager usage
            async with TestHarness(config) as harness:
                await harness.initialize(test_agent)
                assert harness.agent == test_agent


class TestLoggingFeatures:
    """Test cases for enhanced logging features."""

    @pytest.mark.asyncio
    async def test_file_logging_setup(self, tmp_path):
        """Test that file logging is properly configured."""
        log_file = tmp_path / "test.log"

        config = TestHarnessConfig(server_url="http://test:8000", log_file=log_file, log_level="INFO")

        harness = TestHarness(config)

        # Check that the logger is properly configured
        assert hasattr(harness, "logger")
        assert len(harness.logger.handlers) >= 2  # Console + File handlers

        # Check that log file is created
        harness.logger.info("Test log message")
        assert log_file.exists()

        # Check log content
        with open(log_file, "r") as f:
            content = f.read()
            assert "Test log message" in content

    @pytest.mark.asyncio
    async def test_structured_logging_setup(self, tmp_path):
        """Test that structured JSON logging works correctly."""
        log_file = tmp_path / "structured.jsonl"

        config = TestHarnessConfig(
            server_url="http://test:8000", log_file=log_file, log_structured=True, log_level="INFO"
        )

        harness = TestHarness(config)

        # Log a message with extra data
        harness.logger.info(
            "Test structured message", extra={"session_id": "test-123", "step_number": 5, "event_type": "test_event"}
        )

        assert log_file.exists()

        # Check that the log content is valid JSON
        with open(log_file, "r") as f:
            lines = f.readlines()

        # Find the structured log entry we're interested in
        structured_entry = None
        for line in lines:
            try:
                import json

                log_entry = json.loads(line.strip())
                if log_entry.get("message") == "Test structured message":
                    structured_entry = log_entry
                    break
            except json.JSONDecodeError:
                continue

        assert structured_entry is not None, "Structured log entry not found"
        assert structured_entry["message"] == "Test structured message"
        assert structured_entry["session_id"] == "test-123"
        assert structured_entry["step_number"] == 5
        assert structured_entry["event_type"] == "test_event"
        assert "timestamp" in structured_entry
        assert structured_entry["level"] == "INFO"

    @pytest.mark.asyncio
    async def test_log_directory_creation(self, tmp_path):
        """Test that log directories are automatically created."""
        log_file = tmp_path / "subdir" / "nested" / "test.log"

        config = TestHarnessConfig(server_url="http://test:8000", log_file=log_file, log_level="INFO")

        harness = TestHarness(config)
        harness.logger.info("Test message")

        # Directory should be created automatically
        assert log_file.parent.exists()
        assert log_file.exists()

    @pytest.mark.asyncio
    async def test_enhanced_step_logging(self, test_agent, tmp_path):
        """Test that step execution includes structured logging data."""
        log_file = tmp_path / "step_test.jsonl"

        config = TestHarnessConfig(
            server_url="http://test:8000", log_file=log_file, log_structured=True, log_level="INFO", max_steps=1
        )

        # Mock the ServerClient completely
        with patch("saber.client.test_harness.ServerClient") as MockServerClient:
            mock_server = AsyncMock()
            mock_server.create_session = AsyncMock(return_value="test-session")
            mock_server.start_episode = AsyncMock(
                return_value=EpisodeInfo(episode_id="episode-123", task_id="task-456", message="Episode started")
            )
            mock_server.get_current_task = AsyncMock(
                return_value=TaskInfo(
                    task_id="task-456", title="Test Task", description="Test description", current_subtask="subtask-1"
                )
            )
            mock_server.get_policy = AsyncMock(
                return_value=PolicyInfo(
                    domain="test", available_commands=["ls"], guidelines="Test guidelines", constraints=[]
                )
            )
            mock_server.execute_step = AsyncMock(
                return_value=StepResponse(
                    success=True, output="test output", done=True, error=None, info={}  # Complete after one step
                )
            )
            mock_server.close_session = AsyncMock()
            mock_server.health_check = AsyncMock(return_value={"status": "healthy"})

            MockServerClient.return_value = mock_server

            async with TestHarness(config) as harness:
                await harness.initialize(test_agent)
                results = await harness.run_test("test_task")

                # Check that we got results
                assert results["session_id"] == "test-session"
                assert results["episode_id"] == "episode-123"

                # Check the log file contains structured data
                assert log_file.exists()

                with open(log_file, "r") as f:
                    lines = f.readlines()

                # Find log entries with session context
                structured_logs = []
                for line in lines:
                    try:
                        import json

                        log_entry = json.loads(line.strip())
                        if "session_id" in log_entry:
                            structured_logs.append(log_entry)
                    except json.JSONDecodeError:
                        continue

                # Should have logs with session context
                assert len(structured_logs) > 0

                # Check that session_id and episode_id are logged
                session_logs = [log for log in structured_logs if log.get("session_id") == "test-session"]
                assert len(session_logs) > 0

                episode_logs = [log for log in structured_logs if log.get("episode_id") == "episode-123"]
                assert len(episode_logs) > 0
