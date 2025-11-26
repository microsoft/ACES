"""Tests for SABER domain server controller."""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
from inspect_ai._util.error import PrerequisiteError

from saber.inspect_ai.server.server import (
    DomainContext,
    DomainController,
)


class TestDomainContext:
    """Test cases for DomainContext dataclass."""

    def test_domain_context_creation(self):
        """Test creating a DomainContext instance."""
        context = DomainContext(
            domain="test_domain",
            rest_url="http://localhost:8000",
            mcp_url="http://localhost:8001",
            rest_port=8000,
            mcp_port=8001,
            project_slug="test-project",
            domains_root=Path("/test/domains"),
        )

        assert context.domain == "test_domain"
        assert context.rest_url == "http://localhost:8000"
        assert context.mcp_url == "http://localhost:8001"
        assert context.rest_port == 8000
        assert context.mcp_port == 8001
        assert context.project_slug == "test-project"
        assert context.domains_root == Path("/test/domains")


class TestDomainController:
    """Test cases for DomainController class."""

    @pytest.fixture
    def mock_orchestrator(self):
        """Create a mock DomainOrchestrator."""
        return Mock()

    @pytest.fixture
    def controller(self, mock_orchestrator):
        """Create a DomainController with mocked orchestrator."""
        with patch("saber.inspect_ai.server.server._create_orchestrator", return_value=mock_orchestrator):
            return DomainController(Path("/test/domains"))

    @pytest.mark.asyncio
    async def test_check_running_domain_finds_domain(self, controller, mock_orchestrator):
        """Test check_running_domain when a domain is running."""
        # Setup mock to return running domain
        mock_orchestrator.get_domain_status.return_value = {"running": True}

        # Create fake domain directory structure
        with patch.object(Path, "iterdir") as mock_iterdir:
            mock_dir = Mock(spec=Path)
            mock_dir.is_dir.return_value = True
            mock_dir.name = "test_domain"
            mock_iterdir.return_value = [mock_dir]

            result = await controller.check_running_domain(rest_port=8000, mcp_port=8001)

            assert result == "test_domain"
            mock_orchestrator.get_domain_status.assert_called_once_with("test_domain")

    @pytest.mark.asyncio
    async def test_check_running_domain_no_domain_running(self, controller, mock_orchestrator):
        """Test check_running_domain when no domain is running."""
        # Setup mock to return not running
        mock_orchestrator.get_domain_status.return_value = {"running": False}

        with patch.object(Path, "iterdir") as mock_iterdir:
            mock_dir = Mock(spec=Path)
            mock_dir.is_dir.return_value = True
            mock_dir.name = "test_domain"
            mock_iterdir.return_value = [mock_dir]

            result = await controller.check_running_domain()

            assert result is None

    @pytest.mark.asyncio
    async def test_check_running_domain_handles_errors(self, controller, mock_orchestrator):
        """Test check_running_domain handles errors gracefully."""
        # Setup mock to raise an exception
        mock_orchestrator.get_domain_status.side_effect = Exception("Status check failed")

        with patch.object(Path, "iterdir") as mock_iterdir:
            mock_dir = Mock(spec=Path)
            mock_dir.is_dir.return_value = True
            mock_dir.name = "test_domain"
            mock_iterdir.return_value = [mock_dir]

            result = await controller.check_running_domain()

            # Should return None on error, not raise
            assert result is None

    @pytest.mark.asyncio
    async def test_check_running_domain_skips_hidden_dirs(self, controller, mock_orchestrator):
        """Test that check_running_domain skips hidden directories."""
        with patch.object(Path, "iterdir") as mock_iterdir:
            hidden_dir = Mock(spec=Path)
            hidden_dir.is_dir.return_value = True
            hidden_dir.name = ".hidden"

            normal_dir = Mock(spec=Path)
            normal_dir.is_dir.return_value = True
            normal_dir.name = "visible_domain"

            mock_iterdir.return_value = [hidden_dir, normal_dir]
            mock_orchestrator.get_domain_status.return_value = {"running": False}

            await controller.check_running_domain()

            # Should only check visible_domain, not .hidden
            mock_orchestrator.get_domain_status.assert_called_once_with("visible_domain")

    @pytest.mark.asyncio
    async def test_start_domain_success(self, controller, mock_orchestrator):
        """Test successful domain startup."""
        # Mock successful start
        mock_orchestrator.start_domain.return_value = None
        mock_orchestrator.get_project_slug.return_value = "test-project-slug"

        result = await controller.start(
            domain="test_domain",
            rest_port=8000,
            mcp_port=8001,
            log_level="INFO",
        )

        # Verify orchestrator was called correctly
        mock_orchestrator.start_domain.assert_called_once()
        call_kwargs = mock_orchestrator.start_domain.call_args.kwargs
        assert call_kwargs["domain"] == "test_domain"
        assert call_kwargs["rest_port"] == 8000
        assert call_kwargs["mcp_port"] == 8001
        assert call_kwargs["log_level"] == "INFO"
        assert call_kwargs["dry_run"] is False

        # Verify result
        assert isinstance(result, DomainContext)
        assert result.domain == "test_domain"
        assert result.rest_url == "http://localhost:8000"
        assert result.mcp_url == "http://localhost:8001"
        assert result.rest_port == 8000
        assert result.mcp_port == 8001

    @pytest.mark.asyncio
    async def test_start_domain_with_build_options(self, controller, mock_orchestrator):
        """Test domain startup with build options."""
        mock_orchestrator.start_domain.return_value = None
        mock_orchestrator.get_project_slug.return_value = "test-project"

        await controller.start(
            domain="test_domain",
            build="server",
            rebuild="client",
        )

        call_kwargs = mock_orchestrator.start_domain.call_args.kwargs
        assert call_kwargs["build"] == "server"
        assert call_kwargs["rebuild"] == "client"

    @pytest.mark.asyncio
    async def test_start_domain_failure(self, controller, mock_orchestrator):
        """Test domain startup handles errors."""
        # Mock failed start
        mock_orchestrator.start_domain.side_effect = Exception("Startup failed")

        with pytest.raises(PrerequisiteError, match="Failed to start SABER domain"):
            await controller.start(domain="test_domain")

    @pytest.mark.asyncio
    async def test_stop_domain_success(self, controller, mock_orchestrator):
        """Test successful domain shutdown."""
        mock_orchestrator.stop_domain.return_value = None

        await controller.stop(domain="test_domain")

        mock_orchestrator.stop_domain.assert_called_once_with(
            domain="test_domain", dry_run=False
        )

    @pytest.mark.asyncio
    async def test_stop_domain_with_dry_run(self, controller, mock_orchestrator):
        """Test domain shutdown with dry_run mode."""
        # Note: The current implementation doesn't expose dry_run parameter
        # This test verifies the actual parameter passed to orchestrator
        mock_orchestrator.stop_domain.return_value = None

        await controller.stop(domain="test_domain")

        call_kwargs = mock_orchestrator.stop_domain.call_args.kwargs
        assert call_kwargs["dry_run"] is False

    @pytest.mark.asyncio
    async def test_stop_domain_failure(self, controller, mock_orchestrator):
        """Test domain shutdown handles errors."""
        mock_orchestrator.stop_domain.side_effect = Exception("Shutdown failed")

        with pytest.raises(PrerequisiteError, match="Failed to stop SABER domain"):
            await controller.stop(domain="test_domain")

    def test_controller_initialization(self, mock_orchestrator):
        """Test DomainController initialization."""
        with patch("saber.inspect_ai.server.server._create_orchestrator", return_value=mock_orchestrator) as mock_create:
            domains_root = Path("/test/domains")
            controller = DomainController(domains_root)

            assert controller._domains_root == domains_root
            assert controller._orchestrator == mock_orchestrator
            mock_create.assert_called_once_with(domains_root)


class TestCreateOrchestrator:
    """Test cases for _create_orchestrator helper function."""

    def test_create_orchestrator_success(self):
        """Test successful orchestrator creation."""
        from saber.inspect_ai.server.server import _create_orchestrator

        # Mock the imports inside _create_orchestrator using sys.modules
        with patch.dict('sys.modules', {
            'saber.domain.orchestrator': Mock(DomainOrchestrator=Mock()),
            'saber.domain.resources': Mock(resolve_compose_file=Mock()),
        }):
            from saber.domain.orchestrator import DomainOrchestrator
            from saber.domain.resources import resolve_compose_file

            # Mock the context manager for resolve_compose_file
            mock_cm = Mock()
            mock_cm.__enter__ = Mock(return_value=Path("/test/compose.yml"))
            mock_cm.__exit__ = Mock(return_value=False)
            resolve_compose_file.return_value = mock_cm

            mock_orchestrator = Mock()
            DomainOrchestrator.return_value = mock_orchestrator

            result = _create_orchestrator(Path("/test/domains"))

            assert result == mock_orchestrator
            DomainOrchestrator.assert_called_once_with(Path("/test/domains"), Path("/test/compose.yml"))

    def test_create_orchestrator_creation_error(self):
        """Test orchestrator creation with other errors."""
        from saber.inspect_ai.server.server import _create_orchestrator

        with patch.dict('sys.modules', {
            'saber.domain.orchestrator': Mock(DomainOrchestrator=Mock()),
            'saber.domain.resources': Mock(resolve_compose_file=Mock()),
        }):
            from saber.domain.orchestrator import DomainOrchestrator
            from saber.domain.resources import resolve_compose_file

            mock_cm = Mock()
            mock_cm.__enter__ = Mock(return_value=Path("/test/compose.yml"))
            mock_cm.__exit__ = Mock(return_value=False)
            resolve_compose_file.return_value = mock_cm

            DomainOrchestrator.side_effect = Exception("Configuration error")

            with pytest.raises(PrerequisiteError, match="Failed to create SABER DomainOrchestrator"):
                _create_orchestrator(Path("/test/domains"))


class TestConvenienceFunctions:
    """Test convenience functions for domain operations."""

    @pytest.mark.asyncio
    async def test_start_domain_convenience(self):
        """Test start_domain convenience function."""
        from saber.inspect_ai.server.server import start_domain

        with patch("saber.inspect_ai.server.server.DomainController") as mock_controller_class:
            mock_controller = AsyncMock()
            mock_context = DomainContext(
                domain="test_domain",
                rest_url="http://localhost:8000",
                mcp_url="http://localhost:8001",
                rest_port=8000,
                mcp_port=8001,
                project_slug="test-project",
                domains_root=Path("/test/domains"),
            )
            mock_controller.start = AsyncMock(return_value=mock_context)
            mock_controller_class.return_value = mock_controller

            result = await start_domain(
                domain="test_domain",
                domains_root=Path("/test/domains"),
                rest_port=8000,
                mcp_port=8001,
                log_level="DEBUG",
                build="server",
                rebuild="client",
            )

            assert result == mock_context
            mock_controller.start.assert_called_once_with(
                domain="test_domain",
                rest_port=8000,
                mcp_port=8001,
                log_level="DEBUG",
                build="server",
                rebuild="client",
            )

    @pytest.mark.asyncio
    async def test_stop_domain_convenience(self):
        """Test stop_domain convenience function."""
        from saber.inspect_ai.server.server import stop_domain

        with patch("saber.inspect_ai.server.server.DomainController") as mock_controller_class:
            mock_controller = AsyncMock()
            mock_controller.stop = AsyncMock()
            mock_controller_class.return_value = mock_controller

            await stop_domain(
                domain="test_domain",
                domains_root=Path("/test/domains"),
            )

            mock_controller.stop.assert_called_once_with("test_domain")
