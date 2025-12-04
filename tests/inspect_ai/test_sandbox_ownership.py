"""Unit tests for SABERSandboxEnvironment ownership transfer updates.

Tests cover:
- Ownership transfer when factory already started server
- Fresh start when no factory server exists (backward compatibility)
- Registry cleanup in task_cleanup
- Removal from active_domains registry
- Exception handling during ownership transfer
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from saber.inspect_ai.saber import SABERSandboxEnvironment, SandboxError
from saber.inspect_ai.core.types import DomainRegistryEntry


class MockSandboxConfig(BaseModel):
    """Mock config for SABER sandbox tests."""
    domain_slug: str
    domains_root: Path
    rest_port: int = 8000
    mcp_port: int = 8001
    cleanup: bool = False


@pytest.fixture(autouse=True)
def clear_registries():
    """Clear both registries before each test."""
    # Clear sandbox registry
    SABERSandboxEnvironment._registry.clear()

    # Clear active domains registry
    from saber.inspect_ai.server.domain_manager import _active_domains, _active_domains_lock
    with _active_domains_lock:
        _active_domains.clear()

    yield

    SABERSandboxEnvironment._registry.clear()
    with _active_domains_lock:
        _active_domains.clear()


@pytest.fixture
def mock_domain_context():
    """Mock DomainContext."""
    from saber.inspect_ai.server import DomainContext

    return DomainContext(
        domain="test_domain",
        rest_url="http://localhost:8000",
        mcp_url="http://localhost:8001",
        rest_port=8000,
        mcp_port=8001,
        project_slug="saber-test_domain",
        domains_root=Path("/test/domains"),
    )


@pytest.fixture
def mock_controller(mock_domain_context):
    """Mock DomainController."""
    controller = AsyncMock()
    controller.start = AsyncMock(return_value=mock_domain_context)
    controller.stop = AsyncMock()
    return controller


class TestOwnershipTransfer:
    """Test ownership transfer from factory to sandbox."""

    @pytest.mark.asyncio
    async def test_ownership_transfer_from_factory(self, mock_controller, mock_domain_context):
        """Test that sandbox reuses server started by factory."""
        # Simulate factory having started the domain
        from saber.inspect_ai.server.domain_manager import _active_domains, _active_domains_lock

        with _active_domains_lock:
            _active_domains["test_domain"] = {
                "owner": "domain_task_test_domain",
                "domain_slug": "test_domain",
                "controller": mock_controller,
                "context": mock_domain_context,
                "ownership": True,
                "rest_port": 8000,
                "mcp_port": 8001,
                "rest_url": "http://localhost:8000",
                "mcp_url": "http://localhost:8001",
            }

        # Sandbox task_init should transfer ownership
        with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.saber.SABERSandboxEnvironment._create_session_for_task') as mock_create_session:
            mock_controller_class.return_value = mock_controller
            mock_create_session.return_value = "test_session_123"

            config = MockSandboxConfig(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
            )
            await SABERSandboxEnvironment.task_init(
                task_name="test_task",
                config=config,
            )

        # Verify controller.start was NOT called (server already running)
        mock_controller.start.assert_not_called()

        # Verify sandbox registry has entry
        assert "test_domain" in SABERSandboxEnvironment._registry
        entry = SABERSandboxEnvironment._registry["test_domain"]
        assert entry.owner == "test_task"
        assert entry.ownership is True
        assert entry.controller == mock_controller

    @pytest.mark.asyncio
    async def test_fresh_start_when_no_factory_server(self, mock_controller, mock_domain_context):
        """Test backward compatibility: sandbox starts server if factory didn't."""
        # No active domain from factory
        from saber.inspect_ai.server.domain_manager import _active_domains
        assert "test_domain" not in _active_domains

        # Sandbox task_init should start server
        with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.saber.SABERSandboxEnvironment._create_session_for_task') as mock_create_session:
            mock_controller_class.return_value = mock_controller
            mock_create_session.return_value = "test_session_123"

            config = MockSandboxConfig(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
            )
            await SABERSandboxEnvironment.task_init(
                task_name="test_task",
                config=config,
            )

        # Verify controller.start WAS called
        mock_controller.start.assert_called_once()

        # Verify sandbox registry has entry
        assert "test_domain" in SABERSandboxEnvironment._registry
        entry = SABERSandboxEnvironment._registry["test_domain"]
        assert entry.owner == "test_task"
        assert entry.ownership is True

    @pytest.mark.asyncio
    async def test_concurrent_sandbox_init_blocked(self, mock_controller, mock_domain_context):
        """Test that concurrent sandbox init is blocked."""
        # First task initializes
        with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class, \
             patch('saber.inspect_ai.saber.SABERSandboxEnvironment._create_session_for_task') as mock_create_session:
            mock_controller_class.return_value = mock_controller
            mock_create_session.return_value = "test_session_123"

            config = MockSandboxConfig(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
            )
            await SABERSandboxEnvironment.task_init(
                task_name="task1",
                config=config,
            )

        # Second task attempt should fail
        config2 = MockSandboxConfig(
            domain_slug="test_domain",
            domains_root=Path("/test/domains"),
        )
        with pytest.raises(SandboxError) as exc_info:
            await SABERSandboxEnvironment.task_init(
                task_name="task2",
                config=config2,
            )

        assert "already owned" in str(exc_info.value)
        assert "task1" in str(exc_info.value)


class TestTaskCleanup:
    """Test task_cleanup with active_domains registry removal."""

    @pytest.mark.asyncio
    async def test_cleanup_removes_from_both_registries(self, mock_controller, mock_domain_context):
        """Test that cleanup removes from both sandbox and active_domains registries."""
        from saber.inspect_ai.server.domain_manager import _active_domains, _active_domains_lock

        # Populate sandbox registry
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=mock_controller,
            context=mock_domain_context,
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        # Populate active_domains registry
        with _active_domains_lock:
            _active_domains["test_domain"] = {
                "owner": "domain_task_test_domain",
                "controller": mock_controller,
            }

        # Cleanup
        config = MockSandboxConfig(
            domain_slug="test_domain",
            domains_root=Path("/test/domains"),
            cleanup=True,
        )
        await SABERSandboxEnvironment.task_cleanup("test_task", config, cleanup=True)

        # Verify removed from sandbox registry
        assert "test_domain" not in SABERSandboxEnvironment._registry

        # Verify removed from active_domains registry
        with _active_domains_lock:
            assert "test_domain" not in _active_domains

        # Verify controller.stop was called
        mock_controller.stop.assert_called_once_with("test_domain")

    @pytest.mark.asyncio
    async def test_cleanup_handles_missing_sandbox_entry(self, mock_controller):
        """Test cleanup handles missing sandbox registry entry gracefully."""
        from saber.inspect_ai.server.domain_manager import _active_domains, _active_domains_lock

        # Only populate active_domains (simulate partial failure)
        with _active_domains_lock:
            _active_domains["test_domain"] = {
                "owner": "domain_task_test_domain",
                "controller": mock_controller,
            }

        # Cleanup should not raise
        config = MockSandboxConfig(
            domain_slug="test_domain",
            domains_root=Path("/test/domains"),
            cleanup=True,
        )
        await SABERSandboxEnvironment.task_cleanup("test_task", config, cleanup=True)

        # Verify removed from active_domains
        with _active_domains_lock:
            assert "test_domain" not in _active_domains

    @pytest.mark.asyncio
    async def test_cleanup_without_stop(self, mock_controller, mock_domain_context):
        """Test cleanup with cleanup=False doesn't stop server."""
        # Populate sandbox registry
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=mock_controller,
            context=mock_domain_context,
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        # Cleanup without stopping
        config = MockSandboxConfig(
            domain_slug="test_domain",
            domains_root=Path("/test/domains"),
            cleanup=False,
        )
        await SABERSandboxEnvironment.task_cleanup("test_task", config, cleanup=False)

        # Verify NOT removed from registry (preserved for potential retry)
        assert "test_domain" in SABERSandboxEnvironment._registry

        # Verify controller.stop was NOT called
        mock_controller.stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_cleanup_handles_stop_failure(self, mock_controller, mock_domain_context):
        """Test cleanup handles controller.stop failure gracefully."""
        # Setup controller to fail on stop
        mock_controller.stop = AsyncMock(side_effect=Exception("Stop failed"))

        # Populate sandbox registry
        SABERSandboxEnvironment._registry["test_domain"] = DomainRegistryEntry(
            domain_slug="test_domain",
            owner="test_task",
            controller=mock_controller,
            context=mock_domain_context,
            ownership=True,
            rest_port=8000,
            mcp_port=8001,
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            session_id="session_123",
        )

        # Cleanup should not raise
        config = MockSandboxConfig(
            domain_slug="test_domain",
            domains_root=Path("/test/domains"),
            cleanup=True,
        )
        await SABERSandboxEnvironment.task_cleanup("test_task", config, cleanup=True)

        # Verify removed from registry despite stop failure
        assert "test_domain" not in SABERSandboxEnvironment._registry


class TestTaskInitCleanupOnFailure:
    """Test task_init cleanup on failure paths."""

    @pytest.mark.asyncio
    async def test_cleanup_on_controller_start_failure(self, mock_controller):
        """Test that task_init cleans up registry on controller.start failure."""
        # Setup controller to fail
        mock_controller.start = AsyncMock(side_effect=Exception("Start failed"))
        mock_controller.stop = AsyncMock()

        with patch('saber.inspect_ai.saber.DomainController') as mock_controller_class:
            mock_controller_class.return_value = mock_controller

            # task_init should fail
            config = MockSandboxConfig(
                domain_slug="test_domain",
                domains_root=Path("/test/domains"),
            )
            with pytest.raises(SandboxError):
                await SABERSandboxEnvironment.task_init(
                    task_name="test_task",
                    config=config,
                )

        # Verify registry is clean
        assert "test_domain" not in SABERSandboxEnvironment._registry

        # Verify stop was called for cleanup
        mock_controller.stop.assert_called_once()
