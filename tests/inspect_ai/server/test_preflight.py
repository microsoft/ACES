"""Tests for preflight health checks."""

import asyncio
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from inspect_ai._util.error import PrerequisiteError

from saber.inspect_ai.server.preflight import run_preflight_check


@pytest.fixture
def mock_process():
    """Create mock subprocess."""
    process = MagicMock()
    process.returncode = None
    process.wait = AsyncMock(return_value=0)
    process.send_signal = MagicMock()
    process.terminate = MagicMock()
    process.kill = MagicMock()
    return process


class TestRunPreflightCheck:
    """Test run_preflight_check function."""

    @pytest.mark.asyncio
    async def test_preflight_success(self, mock_process):
        """Test successful preflight check."""
        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            # Should not raise
            await run_preflight_check(
                domain_slug="test-domain",
                domains_root=Path("/test/domains"),
                concurrency=8,
                timeout=180,
            )

        # Verify subprocess was created with correct arguments
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        cmd = call_args[0]

        assert "preflight" in cmd
        assert "test-domain" in cmd
        assert "-c" in cmd
        assert "8" in cmd
        assert "--timeout" in cmd
        assert "180" in cmd
        assert "--verbose" in cmd

    @pytest.mark.asyncio
    async def test_preflight_failure(self, mock_process):
        """Test preflight check failure."""
        mock_process.wait = AsyncMock(return_value=1)  # Non-zero exit code

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with pytest.raises(PrerequisiteError) as exc_info:
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        assert "Preflight check failed" in str(exc_info.value)
        assert "exit code: 1" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_preflight_custom_concurrency_and_timeout(self, mock_process):
        """Test preflight with custom concurrency and timeout."""
        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            await run_preflight_check(
                domain_slug="my-domain",
                domains_root=Path("/custom/path"),
                concurrency=16,
                timeout=300,
            )

        call_args = mock_create.call_args[0]
        assert "16" in call_args
        assert "300" in call_args
        assert "my-domain" in call_args

    @pytest.mark.asyncio
    async def test_preflight_file_not_found(self):
        """Test handling of missing saber-domain CLI."""
        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.side_effect = FileNotFoundError("Command not found")

            with pytest.raises(PrerequisiteError) as exc_info:
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        assert "Failed to run saber-domain preflight command" in str(exc_info.value)
        assert "Ensure SABER is properly installed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_preflight_cancelled_graceful_shutdown(self, mock_process):
        """Test graceful shutdown on cancellation."""
        # Simulate user hitting Ctrl+C
        wait_side_effects = [asyncio.CancelledError(), None]  # First raises, second succeeds
        mock_process.wait = AsyncMock(side_effect=wait_side_effects)

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with patch("saber.inspect_ai.server.preflight.asyncio.wait_for") as mock_wait_for:
                # wait_for succeeds (graceful shutdown works)
                mock_wait_for.return_value = None

                with pytest.raises(PrerequisiteError) as exc_info:
                    await run_preflight_check(
                        domain_slug="test-domain",
                        domains_root=Path("/test/domains"),
                    )

        assert "cancelled by user" in str(exc_info.value)
        # Verify SIGINT was sent
        mock_process.send_signal.assert_called_with(signal.SIGINT)

    @pytest.mark.asyncio
    async def test_preflight_cancelled_force_terminate(self, mock_process):
        """Test forced termination when graceful shutdown times out."""
        # First wait (during execution) raises CancelledError
        # Second wait (graceful shutdown) times out
        wait_calls = [
            asyncio.CancelledError(),
            asyncio.TimeoutError(),  # Graceful shutdown timeout
            None,  # Force terminate succeeds
        ]
        mock_process.wait = AsyncMock(side_effect=wait_calls)

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with patch("saber.inspect_ai.server.preflight.asyncio.wait_for") as mock_wait_for:
                # First wait_for (graceful) times out, second (force) succeeds
                mock_wait_for.side_effect = [
                    asyncio.TimeoutError(),
                    None,
                ]

                with pytest.raises(PrerequisiteError):
                    await run_preflight_check(
                        domain_slug="test-domain",
                        domains_root=Path("/test/domains"),
                    )

        # Verify both SIGINT and terminate were called
        mock_process.send_signal.assert_called_with(signal.SIGINT)
        mock_process.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_preflight_cancelled_kill_process(self, mock_process):
        """Test killing process when force terminate times out."""
        # First wait raises CancelledError
        mock_process.wait = AsyncMock(side_effect=asyncio.CancelledError())

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with patch("saber.inspect_ai.server.preflight.asyncio.wait_for") as mock_wait_for:
                # All wait_for calls timeout
                mock_wait_for.side_effect = asyncio.TimeoutError()

                # Need to make the final wait() after kill succeed
                final_wait = AsyncMock(return_value=None)
                mock_process.wait = AsyncMock(side_effect=[
                    asyncio.CancelledError(),
                    final_wait(),
                ])

                with pytest.raises(PrerequisiteError):
                    await run_preflight_check(
                        domain_slug="test-domain",
                        domains_root=Path("/test/domains"),
                    )

        # Verify escalation to kill (may be called twice: once in cancellation, once in finally)
        assert mock_process.kill.call_count >= 1

    @pytest.mark.asyncio
    async def test_preflight_cancelled_already_finished(self, mock_process):
        """Test cancellation when process already finished."""
        # First wait raises CancelledError, but process already has returncode
        mock_process.wait = AsyncMock(side_effect=asyncio.CancelledError())
        mock_process.returncode = 0  # Already finished

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with pytest.raises(PrerequisiteError) as exc_info:
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        # Should not send signals to finished process
        mock_process.send_signal.assert_not_called()
        mock_process.terminate.assert_not_called()
        mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_cancellation_error_handling(self, mock_process):
        """Test error handling during cancellation cleanup."""
        wait_side_effects = [asyncio.CancelledError()]
        mock_process.wait = AsyncMock(side_effect=wait_side_effects)
        mock_process.send_signal = MagicMock(side_effect=Exception("Signal error"))

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            # Should still raise PrerequisiteError despite signal error
            with pytest.raises(PrerequisiteError):
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

    @pytest.mark.asyncio
    async def test_preflight_unexpected_error(self, mock_process):
        """Test handling of unexpected errors."""
        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.side_effect = RuntimeError("Unexpected error")

            with pytest.raises(PrerequisiteError) as exc_info:
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        assert "unexpected error" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_preflight_working_directory(self, mock_process):
        """Test correct working directory is used."""
        domains_root = Path("/test/path/to/domains")

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with patch.object(Path, "exists", return_value=True):
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=domains_root,
                )

        # Verify cwd is parent of domains_root
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["cwd"] == domains_root.parent

    @pytest.mark.asyncio
    async def test_preflight_working_directory_no_parent(self, mock_process):
        """Test working directory when parent doesn't exist."""
        domains_root = Path("/nonexistent/domains")

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with patch.object(Path, "exists", return_value=False):
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=domains_root,
                )

        # Should use domains_root itself as cwd
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["cwd"] == domains_root

    @pytest.mark.asyncio
    async def test_preflight_cleanup_on_error(self):
        """Test process cleanup happens even on error."""
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.wait = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_process.kill = MagicMock()

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            with pytest.raises(PrerequisiteError):
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        # Verify process was killed in finally block
        mock_process.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_preflight_cleanup_already_finished(self):
        """Test cleanup doesn't kill already-finished process."""
        mock_process = MagicMock()
        mock_process.returncode = 0  # Already finished
        mock_process.wait = AsyncMock(return_value=0)
        mock_process.kill = MagicMock()

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            await run_preflight_check(
                domain_slug="test-domain",
                domains_root=Path("/test/domains"),
            )

        # Should not kill already-finished process
        mock_process.kill.assert_not_called()

    @pytest.mark.asyncio
    async def test_preflight_cleanup_error_suppressed(self):
        """Test cleanup errors are suppressed."""
        mock_process = MagicMock()
        mock_process.returncode = None
        mock_process.wait = AsyncMock(side_effect=RuntimeError("Test error"))
        mock_process.kill = MagicMock(side_effect=Exception("Kill error"))

        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            # Should raise original error, not cleanup error
            with pytest.raises(PrerequisiteError) as exc_info:
                await run_preflight_check(
                    domain_slug="test-domain",
                    domains_root=Path("/test/domains"),
                )

        assert "Test error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_preflight_stdout_stderr_passthrough(self, mock_process):
        """Test stdout/stderr are passed through to terminal."""
        with patch("saber.inspect_ai.server.preflight.asyncio.create_subprocess_exec") as mock_create:
            mock_create.return_value = mock_process

            await run_preflight_check(
                domain_slug="test-domain",
                domains_root=Path("/test/domains"),
            )

        # Verify stdout and stderr are None (passed through)
        call_kwargs = mock_create.call_args[1]
        assert call_kwargs["stdout"] is None
        assert call_kwargs["stderr"] is None
