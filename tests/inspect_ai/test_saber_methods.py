"""Additional tests for SABERSandboxEnvironment methods to increase coverage.

Tests focus on:
- Unsupported methods (exec, read_file, write_file, connection)
- _deserialize_benchmark_task
- config_files, config_deserialize
- _get_episode_semaphore
- Error paths and edge cases
"""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from pydantic import create_model, ConfigDict

from saber.inspect_ai.saber import SABERSandboxEnvironment, SandboxError
from saber.inspect_ai.core.types import DomainRegistryEntry
from saber.models import (
    BenchmarkTask,
    SingleEpisodeTask,
    OrchestratedTask,
    TaskExecutionMode,
    MetadataKeys,
    OrchestrationStrategy,
    SubTaskDefinition,
)


class TestUnsupportedMethods:
    """Test that unsupported methods raise NotImplementedError."""

    @pytest.mark.asyncio
    async def test_connection_raises_not_implemented(self):
        """Test that connection() raises NotImplementedError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        with pytest.raises(NotImplementedError, match="connection.*not supported"):
            await instance.connection()

    @pytest.mark.asyncio
    async def test_exec_raises_not_implemented(self):
        """Test that exec() raises NotImplementedError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        with pytest.raises(NotImplementedError, match="exec.*not supported"):
            await instance.exec("ls")

    @pytest.mark.asyncio
    async def test_read_file_raises_not_implemented(self):
        """Test that read_file() raises NotImplementedError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        with pytest.raises(NotImplementedError, match="read_file.*not supported"):
            await instance.read_file("/tmp/file.txt")

    @pytest.mark.asyncio
    async def test_write_file_raises_not_implemented(self):
        """Test that write_file() raises NotImplementedError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        with pytest.raises(NotImplementedError, match="write_file.*not supported"):
            await instance.write_file("/tmp/file.txt", "content")


class TestConfigMethods:
    """Test config_files and config_deserialize methods."""

    def test_config_files_returns_empty_list(self):
        """Test that config_files returns empty list."""
        result = SABERSandboxEnvironment.config_files()
        assert result == []

    def test_config_deserialize_creates_model(self):
        """Test that config_deserialize creates a valid BaseModel."""
        config_dict = {
            "domain_slug": "test_domain",
            "domains_root": Path("/tmp/domains"),
            "rest_port": 8000,
            "mcp_port": 8001,
        }

        config = SABERSandboxEnvironment.config_deserialize(config_dict)

        assert config.domain_slug == "test_domain"
        assert config.domains_root == Path("/tmp/domains")
        assert config.rest_port == 8000
        assert config.mcp_port == 8001

    def test_config_deserialize_with_defaults(self):
        """Test that config_deserialize uses default values."""
        config_dict = {
            "domain_slug": "test_domain",
            "domains_root": Path("/tmp/domains"),
        }

        config = SABERSandboxEnvironment.config_deserialize(config_dict)

        assert config.rest_port == 8000  # Default
        assert config.mcp_port == 8001  # Default
        assert config.cleanup is False  # Default


class TestDeserializeBenchmarkTask:
    """Test _deserialize_benchmark_task method."""

    def test_deserialize_single_episode_task(self):
        """Test deserializing SingleEpisodeTask."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        data = {
            "task_type": TaskExecutionMode.SINGLE.value,
            "benchmark_task_id": "task1",
            "task_id": "task1",
            "domain": "test_domain",
            "title": "Test",
            "description": "Test desc",
            "episode_attempts": 1,
            "max_steps": 10,
            "instruction_prompt": "Do it",
            "assistant_prompt": "OK",
            "submit_prompt": "Submit",
        }

        task = instance._deserialize_benchmark_task(data)

        assert isinstance(task, SingleEpisodeTask)
        assert task.benchmark_task_id == "task1"

    def test_deserialize_orchestrated_task(self):
        """Test deserializing OrchestratedTask."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        sub_task_data = {
            "role": "role1",
            "task_id": "task1",
            "domain": "test_domain",
            "title": "Sub Task",
            "description": "Sub desc",
            "order": 1,
            "depends_on_role": None,
            "episode_attempts": 1,
            "max_steps": 10,
            "instruction_prompt": "Inst",
            "assistant_prompt": "Asst",
            "submit_prompt": "Sub",
        }

        data = {
            "task_type": TaskExecutionMode.ORCHESTRATED.value,
            "benchmark_task_id": "orch1",
            "episode_attempts": 1,
            "orchestration_strategy": OrchestrationStrategy.SEQUENTIAL_PAIRED.value,
            "sub_tasks": [sub_task_data],
        }

        task = instance._deserialize_benchmark_task(data)

        assert isinstance(task, OrchestratedTask)
        assert task.benchmark_task_id == "orch1"

    def test_deserialize_with_execution_mode_fallback(self):
        """Test deserializing using execution_mode instead of task_type."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        data = {
            MetadataKeys.EXECUTION_MODE.value: TaskExecutionMode.SINGLE.value,
            "benchmark_task_id": "task1",
            "task_id": "task1",
            "domain": "test_domain",
            "title": "Test",
            "description": "Test desc",
            "episode_attempts": 1,
            "max_steps": 10,
            "instruction_prompt": "Do it",
            "assistant_prompt": "OK",
            "submit_prompt": "Submit",
        }

        task = instance._deserialize_benchmark_task(data)

        assert isinstance(task, SingleEpisodeTask)

    def test_deserialize_missing_task_type_raises_error(self):
        """Test that missing task_type/execution_mode raises ValueError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        data = {
            "benchmark_task_id": "task1",
            "task_id": "task1",
        }

        with pytest.raises(ValueError, match="Missing.*task_type.*execution_mode"):
            instance._deserialize_benchmark_task(data)

    def test_deserialize_unknown_task_type_raises_error(self):
        """Test that unknown task_type raises ValueError."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )

        data = {
            "task_type": "unknown_type",
            "benchmark_task_id": "task1",
        }

        with pytest.raises(ValueError, match="Unknown task type"):
            instance._deserialize_benchmark_task(data)


class TestGetEpisodeSemaphore:
    """Test _get_episode_semaphore class method."""

    def test_get_episode_semaphore_returns_none_when_unlimited(self):
        """Test that semaphore is None when max_concurrent_episodes is None."""
        SABERSandboxEnvironment._max_concurrent_episodes = None
        SABERSandboxEnvironment._episode_semaphore = None

        semaphore = SABERSandboxEnvironment._get_episode_semaphore()

        assert semaphore is None

    def test_get_episode_semaphore_returns_none_when_zero(self):
        """Test that semaphore is None when max_concurrent_episodes is 0."""
        SABERSandboxEnvironment._max_concurrent_episodes = 0
        SABERSandboxEnvironment._episode_semaphore = None

        semaphore = SABERSandboxEnvironment._get_episode_semaphore()

        assert semaphore is None

    def test_get_episode_semaphore_creates_semaphore(self):
        """Test that semaphore is created with configured limit."""
        import asyncio

        SABERSandboxEnvironment._max_concurrent_episodes = 5
        SABERSandboxEnvironment._episode_semaphore = None

        semaphore = SABERSandboxEnvironment._get_episode_semaphore()

        assert semaphore is not None
        assert isinstance(semaphore, asyncio.Semaphore)
        # Reset for other tests
        SABERSandboxEnvironment._episode_semaphore = None

    def test_get_episode_semaphore_reuses_existing(self):
        """Test that existing semaphore is reused."""
        import asyncio

        SABERSandboxEnvironment._max_concurrent_episodes = 3
        SABERSandboxEnvironment._episode_semaphore = asyncio.Semaphore(3)
        existing_semaphore = SABERSandboxEnvironment._episode_semaphore

        semaphore = SABERSandboxEnvironment._get_episode_semaphore()

        assert semaphore is existing_semaphore
        # Reset for other tests
        SABERSandboxEnvironment._episode_semaphore = None


class TestDefaultConcurrency:
    """Test default_concurrency class method."""

    def test_default_concurrency_returns_none(self):
        """Test that default_concurrency always returns None."""
        result = SABERSandboxEnvironment.default_concurrency()
        assert result is None


class TestClearStaleOwnership:
    """Test clear_stale_ownership class method."""

    def test_clear_stale_ownership_domain_not_in_registry(self):
        """Test clearing ownership for domain not in registry."""
        SABERSandboxEnvironment._registry.clear()

        result = SABERSandboxEnvironment.clear_stale_ownership("nonexistent_domain")

        assert result is False

    def test_clear_stale_ownership_removes_entry(self):
        """Test that clear_stale_ownership removes registry entry."""
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=MagicMock(),
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        result = SABERSandboxEnvironment.clear_stale_ownership("test_domain")

        assert result is True
        assert "test_domain" not in SABERSandboxEnvironment._registry

    def test_clear_stale_ownership_force_flag(self):
        """Test that force flag is logged correctly."""
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=MagicMock(),
            context=MagicMock(),
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        result = SABERSandboxEnvironment.clear_stale_ownership("test_domain", force=True)

        assert result is True
        assert "test_domain" not in SABERSandboxEnvironment._registry


class TestTaskInitEnvironment:
    """Test task_init_environment class method."""

    @pytest.mark.asyncio
    async def test_task_init_environment_returns_empty_dict(self):
        """Test that task_init_environment returns empty dict."""
        result = await SABERSandboxEnvironment.task_init_environment(None, {})

        assert result == {}


class TestSampleInitValidation:
    """Test sample_init input validation."""

    @pytest.mark.asyncio
    async def test_sample_init_missing_task_id_raises_error(self):
        """Test that sample_init raises ValueError when task_id missing."""
        SABERConfig = create_model(
            "SABERConfig",
            domain_slug=(str, ...),
            domains_root=(Path, ...),
            rest_port=(int, 8000),
            mcp_port=(int, 8001),
            compose_template_path=(Path | None, None),
            cleanup=(bool, False),
            max_concurrent_episodes=(int | None, None),
            __config__=ConfigDict(frozen=True),
        )
        config = SABERConfig(
            domain_slug="test_domain",
            domains_root=Path("/tmp/domains"),
        )

        # Metadata missing both task_id and benchmark_task
        metadata = {"sample_id": "sample_123"}

        with pytest.raises(ValueError, match="Missing.*task_id.*benchmark_task"):
            await SABERSandboxEnvironment.sample_init("test_task", config, metadata)


class TestCleanupPartialState:
    """Test _cleanup_partial_state method."""

    @pytest.mark.asyncio
    async def test_cleanup_partial_state_with_no_episode(self):
        """Test cleanup when no episode exists."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )
        instance._episode_id = None
        instance._session_manager = None

        # Should not raise
        await instance._cleanup_partial_state()

    @pytest.mark.asyncio
    async def test_cleanup_partial_state_with_episode_success(self):
        """Test cleanup successfully ends episode."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )
        instance._episode_id = "episode_123"
        instance._session_id = "session_456"

        # Mock the episode manager (new refactored code uses this)
        instance._episode_manager = AsyncMock()
        instance._episode_manager.cleanup_episode = AsyncMock()

        await instance._cleanup_partial_state(interrupted=False)

        instance._episode_manager.cleanup_episode.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_partial_state_with_episode_failure_logged(self):
        """Test that episode cleanup failure is logged but doesn't raise."""
        instance = SABERSandboxEnvironment(
            domain_slug="test",
            domains_root=Path("/tmp"),
        )
        instance._episode_id = "episode_123"
        instance._session_id = "session_456"

        # Mock the episode manager to raise an exception
        instance._episode_manager = AsyncMock()
        instance._episode_manager.cleanup_episode = AsyncMock(side_effect=Exception("End failed"))

        # Should not raise (wrapped in anyio.CancelScope)
        await instance._cleanup_partial_state(interrupted=True)
