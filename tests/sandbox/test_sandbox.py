"""Tests for saber.sandbox — SaberSandboxEnvironment lifecycle."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from inspect_ai.util._sandbox.docker.docker import DockerSandboxEnvironment

from saber.environments.images import (
    ImageBuildResult,
    PreflightBuildError,
    PreflightResult,
    RebuildMode,
)
from saber.sandbox import (
    SaberSandboxEnvironment,
    _cleanup_stale_containers,
    _monitor_startup_progress,
    _parse_docker_timestamp,
    _poll_service_health,
    _remove_conflicting_containers,
    _sanitize_task_name,
    _start_permanent_services,
    _stop_permanent_services,
)


@pytest.fixture(autouse=True)
def _reset_sandbox() -> None:
    """Reset class state between tests."""
    yield  # type: ignore[misc]
    SaberSandboxEnvironment._reset()
    os.environ.pop("SABER_PROJECT", None)


class TestSetKeepPermanent:
    """set_keep_permanent configures keep-alive behavior."""

    def test_set_true(self) -> None:
        SaberSandboxEnvironment.set_keep_permanent(True)
        assert SaberSandboxEnvironment._keep_permanent is True

    def test_set_false(self) -> None:
        SaberSandboxEnvironment.set_keep_permanent(False)
        assert SaberSandboxEnvironment._keep_permanent is False

    def test_default_is_false(self) -> None:
        SaberSandboxEnvironment._reset()
        assert SaberSandboxEnvironment._keep_permanent is False


class TestModuleExports:
    """Module-level __all__ is defined."""

    def test_all_exports(self) -> None:
        import saber.sandbox

        assert hasattr(saber.sandbox, "__all__")
        assert "SaberSandboxEnvironment" in saber.sandbox.__all__


class TestSetPermanentCompose:
    """set_permanent_compose configures class state."""

    def test_sets_compose_path(self) -> None:
        path = Path("/tmp/compose.yml")
        SaberSandboxEnvironment.set_permanent_compose(path, project="test-proj")
        assert SaberSandboxEnvironment._permanent_compose == path
        assert SaberSandboxEnvironment._permanent_project == "test-proj"

    def test_default_project_name(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"))
        assert SaberSandboxEnvironment._permanent_project == "saber-permanent"

    def test_sets_domain_root(self) -> None:
        root = Path("/my/domain")
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), domain_root=root)
        assert SaberSandboxEnvironment._permanent_domain_root == root

    def test_domain_root_defaults_to_none(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"))
        assert SaberSandboxEnvironment._permanent_domain_root is None


class TestTaskInit:
    """task_init starts permanent services."""

    @pytest.mark.asyncio
    async def test_starts_services_when_compose_exists(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock) as mock_start,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_when_no_compose(self) -> None:
        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock) as mock_start,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_start.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_compose_missing(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/nonexistent/compose.yml"))
        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock) as mock_start,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_start.assert_not_called()
            assert "SABER_PROJECT" not in os.environ

    @pytest.mark.asyncio
    async def test_passes_project_directory_as_parent_parent(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock) as mock_start,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            _, kwargs = mock_start.call_args
            assert kwargs["project_directory"] == tmp_path

    @pytest.mark.asyncio
    async def test_uses_explicit_domain_root(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        explicit_root = Path("/explicit/root")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj", domain_root=explicit_root)

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock) as mock_start,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            _, kwargs = mock_start.call_args
            assert kwargs["project_directory"] == explicit_root


class TestTaskCleanup:
    """task_cleanup stops permanent services."""

    @pytest.mark.asyncio
    async def test_stops_when_cleanup_true(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_stop.assert_called_once_with("test-proj")

    @pytest.mark.asyncio
    async def test_skips_when_cleanup_false(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"))
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, False)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_when_no_compose(self) -> None:
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_stop.assert_not_called()


class TestReset:
    """_reset restores defaults."""

    def test_reset_clears_state(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "custom-proj", domain_root=Path("/root"))
        SaberSandboxEnvironment._reset()
        assert SaberSandboxEnvironment._permanent_compose is None
        assert SaberSandboxEnvironment._permanent_project == "saber-permanent"
        assert SaberSandboxEnvironment._permanent_domain_root is None

    def test_reset_clears_keep_permanent(self) -> None:
        SaberSandboxEnvironment.set_keep_permanent(True)
        SaberSandboxEnvironment._reset()
        assert SaberSandboxEnvironment._keep_permanent is False


class TestStartPermanentServices:
    """_start_permanent_services with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_success_first_attempt(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj")

    @pytest.mark.asyncio
    async def test_retry_then_success(self) -> None:
        fail_proc = AsyncMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        ok_proc = AsyncMock()
        ok_proc.returncode = 0
        ok_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[fail_proc, ok_proc]):
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await _start_permanent_services(
                    Path("/tmp/compose.yml"),
                    "test-proj",
                    max_retries=3,
                    base_delay=0.01,
                )

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self) -> None:
        fail_proc = AsyncMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", return_value=fail_proc):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                with pytest.raises(RuntimeError, match="Failed to start"):
                    await _start_permanent_services(
                        Path("/tmp/compose.yml"),
                        "test-proj",
                        max_retries=2,
                        base_delay=0.01,
                    )
                # Sleep should only happen between retries, not after last attempt
                assert mock_sleep.call_count == 1  # max_retries - 1

    @pytest.mark.asyncio
    async def test_project_directory_passed(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _start_permanent_services(
                Path("/tmp/compose.yml"),
                "test-proj",
                project_directory=Path("/domain/root"),
            )
            call_args = mock_exec.call_args[0]
            assert "--project-directory" in call_args
            assert "/domain/root" in call_args

    @pytest.mark.asyncio
    async def test_no_project_directory_when_omitted(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj")
            call_args = mock_exec.call_args[0]
            assert "--project-directory" not in call_args

    @pytest.mark.asyncio
    async def test_includes_build_flag(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj")
            call_args = mock_exec.call_args[0]
            assert "--build" in call_args
            assert call_args.index("--build") > call_args.index("up")

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self) -> None:
        fail_proc = AsyncMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        ok_proc = AsyncMock()
        ok_proc.returncode = 0
        ok_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[fail_proc, fail_proc, ok_proc],
        ):
            with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
                await _start_permanent_services(
                    Path("/tmp/compose.yml"),
                    "test-proj",
                    max_retries=5,
                    base_delay=1.0,
                )
                # First retry: 1.0 * 2^0 = 1.0, Second: 1.0 * 2^1 = 2.0
                delays = [call.args[0] for call in mock_sleep.call_args_list]
                assert delays == [1.0, 2.0]


class TestStopPermanentServices:
    """_stop_permanent_services with mocked subprocess."""

    @pytest.mark.asyncio
    async def test_success(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            await _stop_permanent_services("test-proj")

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            # Should not raise, just warn
            await _stop_permanent_services("test-proj")

    @pytest.mark.asyncio
    async def test_calls_docker_compose_down_with_volumes(self) -> None:
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _stop_permanent_services("my-project")
            call_args = mock_exec.call_args[0]
            assert "docker" in call_args
            assert "compose" in call_args
            assert "down" in call_args
            assert "--volumes" in call_args
            assert "-p" in call_args
            assert "my-project" in call_args


class TestTaskCleanupKeepPermanent:
    """task_cleanup respects keep_permanent flag."""

    @pytest.mark.asyncio
    async def test_skips_stop_when_keep_permanent(self) -> None:
        """When keep_permanent=True, permanent services are NOT stopped."""
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        SaberSandboxEnvironment.set_keep_permanent(True)
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_stop.assert_not_called()

    @pytest.mark.asyncio
    async def test_stops_when_keep_permanent_false(self) -> None:
        """When keep_permanent=False (default), permanent services ARE stopped."""
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        SaberSandboxEnvironment.set_keep_permanent(False)
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_stop.assert_called_once_with("test-proj")

    @pytest.mark.asyncio
    async def test_logs_when_keeping_permanent(self) -> None:
        """When keep_permanent=True, display_progress outputs keeping services message."""
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        SaberSandboxEnvironment.set_keep_permanent(True)
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock),
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
            patch("saber.sandbox.display_progress") as mock_display,
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
        assert any("Keeping permanent services alive" in str(c) for c in mock_display.call_args_list)

    @pytest.mark.asyncio
    async def test_env_var_cleared_even_when_keeping(self) -> None:
        """SABER_PROJECT env var is always cleaned up, even with keep_permanent."""
        os.environ["SABER_PROJECT"] = "test-proj"
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        SaberSandboxEnvironment.set_keep_permanent(True)
        with (
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock),
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            assert "SABER_PROJECT" not in os.environ


class TestSubclassRegistration:
    """SaberSandboxEnvironment is a registered DockerSandboxEnvironment subclass."""

    def test_is_docker_sandbox_subclass(self) -> None:
        assert issubclass(SaberSandboxEnvironment, DockerSandboxEnvironment)

    def test_registered_as_saber(self) -> None:
        from inspect_ai._util.registry import registry_info

        info = registry_info(SaberSandboxEnvironment)
        assert info.type == "sandboxenv"
        assert "saber" in info.name


class TestTaskInitSuperDelegation:
    """task_init delegates to super() with proper ordering."""

    @pytest.mark.asyncio
    async def test_calls_super_task_init(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock) as mock_super_init,
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_super_init.assert_called_once_with("test_task", None)

    @pytest.mark.asyncio
    async def test_super_task_init_called_even_without_permanent(self) -> None:
        with patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock) as mock_super_init:
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_super_init.assert_called_once_with("test_task", None)

    @pytest.mark.asyncio
    async def test_permanent_services_start_before_super(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        order: list[str] = []

        async def track_start(*_args: object, **_kwargs: object) -> None:
            order.append("start_permanent")

        async def track_super(*_args: object, **_kwargs: object) -> None:
            order.append("super_init")

        with (
            patch(
                "saber.sandbox._start_permanent_services",
                side_effect=track_start,
            ),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                side_effect=track_super,
            ),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert order == ["start_permanent", "super_init"]

    @pytest.mark.asyncio
    async def test_task_init_error_recovery_stops_services(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                new_callable=AsyncMock,
                side_effect=RuntimeError("docker broke"),
            ),
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
        ):
            with pytest.raises(RuntimeError, match="docker broke"):
                await SaberSandboxEnvironment.task_init("test_task", None)
            mock_stop.assert_called_once_with("test-proj")

    @pytest.mark.asyncio
    async def test_task_init_error_recovery_clears_env_var(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                new_callable=AsyncMock,
                side_effect=RuntimeError("docker broke"),
            ),
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="docker broke"):
                await SaberSandboxEnvironment.task_init("test_task", None)
            assert "SABER_PROJECT" not in os.environ

    @pytest.mark.asyncio
    async def test_task_init_double_failure_preserves_original(self, tmp_path: Path) -> None:
        """If super().task_init() and _stop_permanent_services both raise,
        the original exception propagates (not the stop failure)."""
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                new_callable=AsyncMock,
                side_effect=RuntimeError("init broke"),
            ),
            patch(
                "saber.sandbox._stop_permanent_services",
                new_callable=AsyncMock,
                side_effect=OSError("stop also broke"),
            ),
        ):
            with pytest.raises(RuntimeError, match="init broke"):
                await SaberSandboxEnvironment.task_init("test_task", None)
            assert "SABER_PROJECT" not in os.environ


class TestTaskCleanupSuperDelegation:
    """task_cleanup delegates to super() with proper ordering."""

    @pytest.mark.asyncio
    async def test_calls_super_task_cleanup(self) -> None:
        with patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock) as mock_super_cleanup:
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_super_cleanup.assert_called_once_with("test_task", None, True)

    @pytest.mark.asyncio
    async def test_cleanup_signature_positional(self) -> None:
        """cleanup is positional, not keyword-only."""
        import inspect

        sig = inspect.signature(SaberSandboxEnvironment.task_cleanup)
        cleanup_param = sig.parameters["cleanup"]
        assert cleanup_param.kind != inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.asyncio
    async def test_permanent_stop_after_super_cleanup(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        order: list[str] = []

        async def track_super(*_args: object, **_kwargs: object) -> None:
            order.append("super_cleanup")

        async def track_stop(*_args: object, **_kwargs: object) -> None:
            order.append("stop_permanent")

        with (
            patch.object(
                DockerSandboxEnvironment,
                "task_cleanup",
                side_effect=track_super,
            ),
            patch(
                "saber.sandbox._stop_permanent_services",
                side_effect=track_stop,
            ),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            assert order == ["super_cleanup", "stop_permanent"]

    @pytest.mark.asyncio
    async def test_permanent_stop_even_if_super_raises(self) -> None:
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        with (
            patch.object(
                DockerSandboxEnvironment,
                "task_cleanup",
                new_callable=AsyncMock,
                side_effect=RuntimeError("cleanup failed"),
            ),
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock) as mock_stop,
        ):
            with pytest.raises(RuntimeError, match="cleanup failed"):
                await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            mock_stop.assert_called_once_with("test-proj")


class TestEnvVarInjection:
    """SABER_PROJECT env var lifecycle."""

    @pytest.mark.asyncio
    async def test_task_init_sets_env_var(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert os.environ["SABER_PROJECT"] == "test-proj"

    @pytest.mark.asyncio
    async def test_env_var_set_before_super_task_init(self, tmp_path: Path) -> None:
        compose_file = tmp_path / "compose" / "test.yml"
        compose_file.parent.mkdir()
        compose_file.write_text("services: {}")
        SaberSandboxEnvironment.set_permanent_compose(compose_file, "test-proj")

        captured_env: dict[str, str | None] = {}

        async def capture_env(*_args: object, **_kwargs: object) -> None:
            captured_env["value"] = os.environ.get("SABER_PROJECT")

        with (
            patch("saber.sandbox._start_permanent_services", new_callable=AsyncMock),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                side_effect=capture_env,
            ),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert captured_env["value"] == "test-proj"

    @pytest.mark.asyncio
    async def test_task_cleanup_clears_env_var(self) -> None:
        os.environ["SABER_PROJECT"] = "test-proj"
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        with (
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
            patch("saber.sandbox._stop_permanent_services", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            assert "SABER_PROJECT" not in os.environ

    @pytest.mark.asyncio
    async def test_env_var_cleared_even_if_stop_fails(self) -> None:
        os.environ["SABER_PROJECT"] = "test-proj"
        SaberSandboxEnvironment.set_permanent_compose(Path("/tmp/c.yml"), "test-proj")
        with (
            patch.object(DockerSandboxEnvironment, "task_cleanup", new_callable=AsyncMock),
            patch(
                "saber.sandbox._stop_permanent_services",
                new_callable=AsyncMock,
                side_effect=RuntimeError("stop failed"),
            ),
        ):
            # Should not raise — stop failure is caught
            await SaberSandboxEnvironment.task_cleanup("test_task", None, True)
            assert "SABER_PROJECT" not in os.environ

    @pytest.mark.asyncio
    async def test_no_env_var_without_permanent_compose(self) -> None:
        with patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock):
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert "SABER_PROJECT" not in os.environ

    def test_reset_clears_env_var(self) -> None:
        os.environ["SABER_PROJECT"] = "test-proj"
        SaberSandboxEnvironment._reset()
        assert "SABER_PROJECT" not in os.environ


class TestSetPreflightConfig:
    """set_preflight_config configures preflight state."""

    def test_sets_preflight_state(self) -> None:
        root = Path("/my/domain")
        rebuild = RebuildMode.all()
        SaberSandboxEnvironment.set_preflight_config(root, rebuild)
        assert SaberSandboxEnvironment._preflight_domain_root == root
        assert SaberSandboxEnvironment._preflight_rebuild == rebuild
        assert SaberSandboxEnvironment._preflight_done is False

    def test_defaults_rebuild_to_none(self) -> None:
        SaberSandboxEnvironment.set_preflight_config(Path("/root"))
        assert SaberSandboxEnvironment._preflight_rebuild is None

    def test_reset_clears_preflight(self) -> None:
        SaberSandboxEnvironment.set_preflight_config(Path("/root"), RebuildMode.all())
        SaberSandboxEnvironment._reset()
        assert SaberSandboxEnvironment._preflight_domain_root is None
        assert SaberSandboxEnvironment._preflight_rebuild is None
        assert SaberSandboxEnvironment._preflight_done is False
        assert SaberSandboxEnvironment._preflight_error is None

    def test_reset_recreates_preflight_lock(self) -> None:
        """_reset() should create a fresh lock to prevent cross-test deadlocks."""
        old_lock = SaberSandboxEnvironment._preflight_lock
        SaberSandboxEnvironment._reset()
        assert SaberSandboxEnvironment._preflight_lock is not old_lock


class TestPreflightInTaskInit:
    """task_init runs preflight before super().task_init()."""

    @pytest.mark.asyncio
    async def test_calls_build_domain_images(self, tmp_path: Path) -> None:
        SaberSandboxEnvironment.set_preflight_config(tmp_path, RebuildMode.all())

        mock_result = PreflightResult(
            results=(ImageBuildResult(name="base", tag="saber/sandbox:latest", action="rebuilt"),),
            domain_slug="test",
        )
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock, return_value=mock_result) as mock_build,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_build.assert_called_once_with(tmp_path, rebuild=RebuildMode.all())

    @pytest.mark.asyncio
    async def test_skips_when_no_preflight_config(self) -> None:
        # No set_preflight_config called
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock) as mock_build,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_build.assert_not_called()

    @pytest.mark.asyncio
    async def test_raises_on_build_failure(self, tmp_path: Path) -> None:
        SaberSandboxEnvironment.set_preflight_config(tmp_path)

        mock_result = PreflightResult(
            results=(ImageBuildResult(name="base", tag="saber/sandbox:latest", action="failed", error="boom"),),
            domain_slug="test",
        )
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock, return_value=mock_result),
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            with pytest.raises(PreflightBuildError, match="PREFLIGHT IMAGE BUILD FAILED"):
                await SaberSandboxEnvironment.task_init("test_task", None)

    @pytest.mark.asyncio
    async def test_build_failure_includes_docker_errors(self, tmp_path: Path) -> None:
        """PreflightBuildError message includes per-image Docker build errors."""
        SaberSandboxEnvironment.set_preflight_config(tmp_path)

        mock_result = PreflightResult(
            results=(
                ImageBuildResult(
                    name="db",
                    tag="saber/db:latest",
                    action="failed",
                    error="COPY failed: file not found",
                ),
                ImageBuildResult(
                    name="web",
                    tag="saber/web:latest",
                    action="failed",
                    error="apt-get install failed",
                ),
            ),
            domain_slug="test",
        )
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock, return_value=mock_result),
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            with pytest.raises(PreflightBuildError) as exc_info:
                await SaberSandboxEnvironment.task_init("test_task", None)
            msg = str(exc_info.value)
            assert "2 image(s) failed to build: db, web" in msg
            assert "COPY failed: file not found" in msg
            assert "apt-get install failed" in msg
            assert "saber/db:latest" in msg
            assert "saber/web:latest" in msg

    @pytest.mark.asyncio
    async def test_subsequent_call_re_raises_stored_error(self, tmp_path: Path) -> None:
        """After preflight failure, subsequent task_init calls re-raise the error."""
        SaberSandboxEnvironment.set_preflight_config(tmp_path)

        mock_result = PreflightResult(
            results=(ImageBuildResult(name="web", tag="web:1", action="failed", error="boom"),),
            domain_slug="test",
        )
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock, return_value=mock_result) as mock_build,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            with pytest.raises(PreflightBuildError, match="PREFLIGHT IMAGE BUILD FAILED"):
                await SaberSandboxEnvironment.task_init("test_task", None)
            # Second call should also raise without calling build_domain_images again
            with pytest.raises(PreflightBuildError, match="PREFLIGHT IMAGE BUILD FAILED"):
                await SaberSandboxEnvironment.task_init("test_task", None)
            assert mock_build.call_count == 1

    @pytest.mark.asyncio
    async def test_runs_only_once(self, tmp_path: Path) -> None:
        """Preflight runs only on first task_init, not subsequent calls."""
        SaberSandboxEnvironment.set_preflight_config(tmp_path)

        mock_result = PreflightResult(
            results=(ImageBuildResult(name="base", tag="saber/sandbox:latest", action="skipped"),),
            domain_slug="test",
        )
        with (
            patch("saber.sandbox.build_domain_images", new_callable=AsyncMock, return_value=mock_result) as mock_build,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert mock_build.call_count == 1

    @pytest.mark.asyncio
    async def test_preflight_before_super(self, tmp_path: Path) -> None:
        """Preflight (build_domain_images) runs BEFORE super().task_init()."""
        call_order: list[str] = []
        SaberSandboxEnvironment.set_preflight_config(tmp_path)

        mock_result = PreflightResult(
            results=(ImageBuildResult(name="base", tag="saber/sandbox:latest", action="skipped"),),
            domain_slug="test",
        )

        async def mock_build(*a: object, **kw: object) -> PreflightResult:
            call_order.append("preflight")
            return mock_result

        async def mock_super_init(*a: object, **kw: object) -> None:
            call_order.append("super_init")

        with (
            patch("saber.sandbox.build_domain_images", side_effect=mock_build),
            patch.object(DockerSandboxEnvironment, "task_init", side_effect=mock_super_init),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)

        assert call_order == ["preflight", "super_init"]


# =========================================================================
# Startup Progress Monitoring
# =========================================================================


class TestPollServiceHealth:
    """_poll_service_health parses docker compose ps JSON output."""

    @pytest.mark.asyncio
    async def test_returns_service_tuples(self) -> None:
        """Single service JSON parsed into (service, state, health)."""
        json_line = '{"Service":"db","State":"running","Health":"starting"}'
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json_line.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert result == [("db", "running", "starting")]

    @pytest.mark.asyncio
    async def test_handles_multi_line_json(self) -> None:
        """Multiple services, one JSON object per line."""
        output = (
            '{"Service":"db","State":"running","Health":"healthy"}\n'
            '{"Service":"web","State":"running","Health":"starting"}\n'
        )
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(output.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert len(result) == 2
        assert result[0] == ("db", "running", "healthy")
        assert result[1] == ("web", "running", "starting")

    @pytest.mark.asyncio
    async def test_returns_empty_on_nonzero_exit(self) -> None:
        """Non-zero exit code returns empty list."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 1
        mock_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_exception(self) -> None:
        """OSError from subprocess returns empty list."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("docker not found"),
        ):
            result = await _poll_service_health("test-proj")

        assert result == []

    @pytest.mark.asyncio
    async def test_returns_empty_on_invalid_json(self) -> None:
        """Invalid JSON output returns empty list."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"not json", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert result == []

    @pytest.mark.asyncio
    async def test_passes_correct_command(self) -> None:
        """Verifies exact docker compose ps command args."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc) as mock_exec:
            await _poll_service_health("my-project")

        args = mock_exec.call_args[0]
        assert args == (
            "docker",
            "compose",
            "-p",
            "my-project",
            "ps",
            "--format",
            "json",
        )

    @pytest.mark.asyncio
    async def test_falls_back_to_name_key(self) -> None:
        """Uses 'Name' key when 'Service' is absent."""
        json_line = '{"Name":"my-container","State":"running","Health":"healthy"}'
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(json_line.encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert result == [("my-container", "running", "healthy")]

    @pytest.mark.asyncio
    async def test_empty_stdout_returns_empty(self) -> None:
        """Empty stdout returns empty list."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            result = await _poll_service_health("test-proj")

        assert result == []


class TestMonitorStartupProgress:
    """_monitor_startup_progress logs waiting services periodically."""

    @pytest.mark.asyncio
    async def test_logs_waiting_services(self) -> None:
        """Non-healthy services are displayed via display_progress."""
        call_count = 0

        async def mock_poll(project: str) -> list[tuple[str, str, str]]:
            nonlocal call_count
            call_count += 1
            return [("db", "running", "starting")]

        async def mock_sleep(seconds: float) -> None:
            if call_count >= 1:
                raise asyncio.CancelledError

        with (
            patch("saber.sandbox._poll_service_health", side_effect=mock_poll),
            patch("saber.sandbox.asyncio.sleep", side_effect=mock_sleep),
            patch("saber.sandbox.display_progress") as mock_display,
        ):
            task = asyncio.create_task(_monitor_startup_progress("test-proj", interval=15.0))
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert any("Waiting for services: db (health: starting)" in str(c) for c in mock_display.call_args_list)

    @pytest.mark.asyncio
    async def test_skips_log_when_all_healthy(self, caplog: pytest.LogCaptureFixture) -> None:
        """No log when all services are running+healthy."""
        call_count = 0

        async def mock_poll(project: str) -> list[tuple[str, str, str]]:
            nonlocal call_count
            call_count += 1
            return [("db", "running", "healthy")]

        async def mock_sleep(seconds: float) -> None:
            if call_count >= 1:
                raise asyncio.CancelledError

        with (
            patch("saber.sandbox._poll_service_health", side_effect=mock_poll),
            patch("saber.sandbox.asyncio.sleep", side_effect=mock_sleep),
            caplog.at_level(logging.INFO, logger="saber.sandbox"),
        ):
            task = asyncio.create_task(_monitor_startup_progress("test-proj", interval=15.0))
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not any("Waiting for services" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_cancellable(self) -> None:
        """Task can be cancelled cleanly with no exception leaks."""
        with patch("saber.sandbox.asyncio.sleep", new_callable=AsyncMock):
            task = asyncio.create_task(_monitor_startup_progress("test-proj", interval=0.01))
            await asyncio.sleep(0)  # yield to event loop
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
            assert task.cancelled()

    @pytest.mark.asyncio
    async def test_respects_interval(self) -> None:
        """Passes configured interval to asyncio.sleep."""
        call_count = 0

        async def mock_poll(project: str) -> list[tuple[str, str, str]]:
            nonlocal call_count
            call_count += 1
            return []

        async def mock_sleep(seconds: float) -> None:
            if call_count >= 1:
                raise asyncio.CancelledError

        with (
            patch("saber.sandbox._poll_service_health", side_effect=mock_poll),
            patch("saber.sandbox.asyncio.sleep", side_effect=mock_sleep) as sleep_mock,
        ):
            task = asyncio.create_task(_monitor_startup_progress("proj", interval=42.0))
            with contextlib.suppress(asyncio.CancelledError):
                await task
            sleep_mock.assert_called_with(42.0)

    @pytest.mark.asyncio
    async def test_continues_on_empty_poll(self, caplog: pytest.LogCaptureFixture) -> None:
        """Empty poll result (error case) doesn't crash the loop."""
        call_count = 0

        async def mock_poll(project: str) -> list[tuple[str, str, str]]:
            nonlocal call_count
            call_count += 1
            return []

        async def mock_sleep(seconds: float) -> None:
            if call_count >= 2:
                raise asyncio.CancelledError

        with (
            patch("saber.sandbox._poll_service_health", side_effect=mock_poll),
            patch("saber.sandbox.asyncio.sleep", side_effect=mock_sleep),
            caplog.at_level(logging.INFO, logger="saber.sandbox"),
        ):
            task = asyncio.create_task(_monitor_startup_progress("test-proj"))
            with contextlib.suppress(asyncio.CancelledError):
                await task

        # Should have polled twice before cancel
        assert call_count == 2
        assert not any("Waiting for services" in r.message for r in caplog.records)


class TestStartPermanentServicesProgress:
    """Progress monitoring integration in _start_permanent_services."""

    @pytest.mark.asyncio
    async def test_logs_starting_message(self) -> None:
        """REQ-001: display_progress messages before and after docker compose."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("saber.sandbox._monitor_startup_progress", new_callable=AsyncMock),
            patch("saber.sandbox.display_progress") as mock_display,
        ):
            await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj")

        messages = [str(c) for c in mock_display.call_args_list]
        assert any("Starting permanent services" in m for m in messages)
        assert any("Permanent services started" in m for m in messages)

    @pytest.mark.asyncio
    async def test_monitor_task_launched(self) -> None:
        """REQ-002: _monitor_startup_progress is called during startup."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"ok", b""))

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_proc),
            patch("saber.sandbox._monitor_startup_progress", new_callable=AsyncMock) as mock_monitor,
        ):
            await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj")

        mock_monitor.assert_called_once_with("test-proj")

    @pytest.mark.asyncio
    async def test_monitor_cancelled_on_failure(self) -> None:
        """REQ-004: Monitor task is cancelled even when startup fails."""
        fail_proc = AsyncMock()
        fail_proc.returncode = 1
        fail_proc.communicate = AsyncMock(return_value=(b"", b"error"))

        with (
            patch("asyncio.create_subprocess_exec", return_value=fail_proc),
            patch("saber.sandbox.asyncio.sleep", new_callable=AsyncMock),
            patch("saber.sandbox._monitor_startup_progress", new_callable=AsyncMock) as mock_monitor,
        ):
            with pytest.raises(RuntimeError, match="Failed to start"):
                await _start_permanent_services(Path("/tmp/compose.yml"), "test-proj", max_retries=1)

        mock_monitor.assert_called()


class TestSanitizeTaskName:
    """_sanitize_task_name matches inspect_ai's task_project_name logic."""

    def test_simple_name(self) -> None:
        assert _sanitize_task_name("excytin") == "excytin"

    def test_uppercase_lowered(self) -> None:
        assert _sanitize_task_name("ExCyTiN") == "excytin"

    def test_special_chars_replaced(self) -> None:
        # "My.Complex/Task_Name!" -> "my-complex-task_name-" -> first 12 -> "my-complex-t"
        assert _sanitize_task_name("My.Complex/Task_Name!") == "my-complex-t"

    def test_trailing_underscore_stripped(self) -> None:
        assert _sanitize_task_name("abcdefghijk_") == "abcdefghijk"

    def test_collapsed_dashes(self) -> None:
        assert _sanitize_task_name("a--b---c") == "a-b-c"

    def test_truncated_to_12_chars(self) -> None:
        result = _sanitize_task_name("abcdefghijklmnop")
        assert len(result) <= 12
        assert result == "abcdefghijkl"


class TestParseDockerTimestamp:
    """_parse_docker_timestamp handles Docker's timestamp format."""

    def test_parses_standard_format(self) -> None:
        ts = "2026-03-06 05:58:30 +0000 UTC"
        result = _parse_docker_timestamp(ts)
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 6
        assert result.hour == 5
        assert result.minute == 58
        assert result.second == 30

    def test_returns_utc_aware(self) -> None:
        ts = "2026-03-06 05:58:30 +0000 UTC"
        result = _parse_docker_timestamp(ts)
        assert result.tzinfo is not None

    def test_raises_on_garbage(self) -> None:
        with pytest.raises(ValueError):
            _parse_docker_timestamp("not a timestamp")


class TestCleanupStaleContainers:
    """_cleanup_stale_containers removes zombie Docker containers."""

    def _old_timestamp(self, minutes: int = 10) -> str:
        """Create a Docker-formatted timestamp `minutes` in the past."""
        ts = datetime.now(tz=UTC) - timedelta(minutes=minutes)
        return ts.strftime("%Y-%m-%d %H:%M:%S %z") + " UTC"

    def _recent_timestamp(self, seconds: int = 30) -> str:
        """Create a Docker-formatted timestamp `seconds` in the past."""
        ts = datetime.now(tz=UTC) - timedelta(seconds=seconds)
        return ts.strftime("%Y-%m-%d %H:%M:%S %z") + " UTC"

    @pytest.mark.asyncio
    async def test_removes_old_containers_only(self) -> None:
        """Only containers older than min_age_seconds are removed."""
        old_ts = self._old_timestamp(minutes=10)
        recent_ts = self._recent_timestamp(seconds=30)
        ps_output = f"abc123 {old_ts}\ndef456 {recent_ts}\n".encode()

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[find_proc, rm_proc]) as mock_exec:
            await _cleanup_stale_containers("excytin")

        # Only the old container should be removed
        rm_call = mock_exec.call_args_list[1]
        rm_args = rm_call[0]
        assert "abc123" in rm_args
        assert "def456" not in rm_args

    @pytest.mark.asyncio
    async def test_removes_all_old_containers(self) -> None:
        """All containers exceeding the age threshold are removed."""
        old_ts1 = self._old_timestamp(minutes=5)
        old_ts2 = self._old_timestamp(minutes=20)
        ps_output = f"aaa111 {old_ts1}\nbbb222 {old_ts2}\n".encode()

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[find_proc, rm_proc]) as mock_exec:
            await _cleanup_stale_containers("excytin")

        rm_call = mock_exec.call_args_list[1]
        rm_args = rm_call[0]
        assert "aaa111" in rm_args
        assert "bbb222" in rm_args

    @pytest.mark.asyncio
    async def test_skips_all_young_containers(self) -> None:
        """When all containers are younger than threshold, none are removed."""
        recent_ts = self._recent_timestamp(seconds=30)
        ps_output = f"abc123 {recent_ts}\n".encode()

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        with patch("asyncio.create_subprocess_exec", return_value=find_proc) as mock_exec:
            await _cleanup_stale_containers("excytin")

        # Only the find call; no rm call
        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_no_containers_found_skips_removal(self) -> None:
        """When no stale containers exist, docker rm is not called."""
        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=find_proc) as mock_exec:
            await _cleanup_stale_containers("excytin")

        assert mock_exec.call_count == 1

    @pytest.mark.asyncio
    async def test_errors_are_logged_not_raised(self) -> None:
        """Exceptions during cleanup are caught and logged, not propagated."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("docker not found"),
        ):
            await _cleanup_stale_containers("excytin")

    @pytest.mark.asyncio
    async def test_errors_are_logged_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Exceptions during cleanup produce a warning log."""
        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=OSError("docker not found"),
        ):
            with caplog.at_level(logging.WARNING):
                await _cleanup_stale_containers("excytin")

        assert any("stale" in r.message.lower() or "cleanup" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_uses_correct_name_filter(self) -> None:
        """Docker ps filter uses the sanitized task name pattern."""
        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=find_proc) as mock_exec:
            await _cleanup_stale_containers("excytin")

        find_args = mock_exec.call_args_list[0][0]
        filter_args = [a for a in find_args if a.startswith("name=")]
        assert filter_args == ["name=inspect-excytin-"]

    @pytest.mark.asyncio
    async def test_uses_created_and_dead_status_filters(self) -> None:
        """Docker ps is called with both 'created' and 'dead' status filters."""
        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", return_value=find_proc) as mock_exec:
            await _cleanup_stale_containers("excytin")

        find_args = mock_exec.call_args_list[0][0]
        status_filters = [a for a in find_args if a.startswith("status=")]
        assert "status=created" in status_filters
        assert "status=dead" in status_filters

    @pytest.mark.asyncio
    async def test_logs_warning_when_stale_found(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning is logged when stale containers are found and removed."""
        old_ts = self._old_timestamp(minutes=10)
        ps_output = f"abc123 {old_ts}\n".encode()

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[find_proc, rm_proc]):
            with caplog.at_level(logging.WARNING):
                await _cleanup_stale_containers("excytin")

        assert any("stale" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_treated_as_stale(self) -> None:
        """Containers with unparseable timestamps are removed (safe default)."""
        ps_output = b"abc123 NOT_A_TIMESTAMP\n"

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[find_proc, rm_proc]) as mock_exec:
            await _cleanup_stale_containers("excytin")

        rm_call = mock_exec.call_args_list[1]
        assert "abc123" in rm_call[0]

    @pytest.mark.asyncio
    async def test_custom_min_age_seconds(self) -> None:
        """The min_age_seconds parameter overrides the default threshold."""
        # Container is 2 minutes old
        ts = self._old_timestamp(minutes=2)
        ps_output = f"abc123 {ts}\n".encode()

        find_proc = AsyncMock()
        find_proc.returncode = 0
        find_proc.communicate = AsyncMock(return_value=(ps_output, b""))

        # With default 180s threshold, this 2-min container is NOT stale
        with patch("asyncio.create_subprocess_exec", return_value=find_proc) as mock_exec:
            await _cleanup_stale_containers("excytin", min_age_seconds=180)

        assert mock_exec.call_count == 1  # no rm call

        # With a 60s threshold, the same container IS stale
        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[find_proc, rm_proc]) as mock_exec:
            await _cleanup_stale_containers("excytin", min_age_seconds=60)

        assert mock_exec.call_count == 2  # find + rm


class TestTaskInitStaleContainerCleanup:
    """task_init calls _cleanup_stale_containers before super()."""

    @pytest.mark.asyncio
    async def test_cleans_stale_containers_before_super(self) -> None:
        """_cleanup_stale_containers runs before super().task_init()."""
        order: list[str] = []

        async def track_cleanup(*_args: object, **_kwargs: object) -> None:
            order.append("cleanup_stale")

        async def track_super(*_args: object, **_kwargs: object) -> None:
            order.append("super_init")

        with (
            patch(
                "saber.sandbox._cleanup_stale_containers",
                side_effect=track_cleanup,
            ),
            patch.object(
                DockerSandboxEnvironment,
                "task_init",
                side_effect=track_super,
            ),
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            assert "cleanup_stale" in order
            assert "super_init" in order
            assert order.index("cleanup_stale") < order.index("super_init")

    @pytest.mark.asyncio
    async def test_cleanup_called_with_task_name(self) -> None:
        """_cleanup_stale_containers receives the task_name argument."""
        with (
            patch(
                "saber.sandbox._cleanup_stale_containers",
                new_callable=AsyncMock,
            ) as mock_cleanup,
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock),
        ):
            await SaberSandboxEnvironment.task_init("my_task", None)
            mock_cleanup.assert_called_once_with("my_task")

    @pytest.mark.asyncio
    async def test_cleanup_failure_does_not_block_init(self) -> None:
        """If _cleanup_stale_containers raises, task_init still proceeds."""
        with (
            patch(
                "saber.sandbox._cleanup_stale_containers",
                new_callable=AsyncMock,
                side_effect=RuntimeError("cleanup failed"),
            ),
            patch.object(DockerSandboxEnvironment, "task_init", new_callable=AsyncMock) as mock_super,
        ):
            await SaberSandboxEnvironment.task_init("test_task", None)
            mock_super.assert_called_once()


class TestRemoveConflictingContainers:
    """_remove_conflicting_containers validates staleness before removal."""

    _CONFLICT_MSG = (
        'Container inspect-excytin-i6gkber-default-1  Creating\n'
        'Error response from daemon: Conflict. The container name '
        '"/inspect-excytin-i6gkber-default-1" is already in use by '
        'container "f782a09e458bbbd81200918afbd06c96c6155feb04026c326344d0ed1590b405". '
        'You have to remove (or rename) that container to be able to reuse that name.'
    )

    def _old_iso_timestamp(self, minutes: int = 10) -> str:
        """Return a Docker-inspect-style ISO timestamp `minutes` ago."""
        ts = datetime.now(tz=UTC) - timedelta(minutes=minutes)
        return ts.strftime("%Y-%m-%dT%H:%M:%S") + ".000000000Z"

    def _recent_iso_timestamp(self, seconds: int = 30) -> str:
        """Return a Docker-inspect-style ISO timestamp `seconds` ago."""
        ts = datetime.now(tz=UTC) - timedelta(seconds=seconds)
        return ts.strftime("%Y-%m-%dT%H:%M:%S") + ".000000000Z"

    @pytest.mark.asyncio
    async def test_removes_stale_created_container(self) -> None:
        """A container in 'created' state older than threshold is removed."""
        old_ts = self._old_iso_timestamp(minutes=10)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"created {old_ts}".encode(), b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[inspect_proc, rm_proc]) as mock_exec:
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is True
        rm_call = mock_exec.call_args_list[1]
        assert "docker" in rm_call[0]
        assert "rm" in rm_call[0]
        assert "-f" in rm_call[0]

    @pytest.mark.asyncio
    async def test_skips_running_container(self) -> None:
        """A container in 'running' state is NOT removed."""
        old_ts = self._old_iso_timestamp(minutes=10)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"running {old_ts}".encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=inspect_proc) as mock_exec:
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is False
        assert mock_exec.call_count == 1  # inspect only, no rm

    @pytest.mark.asyncio
    async def test_skips_young_container(self) -> None:
        """A container younger than min_age_seconds is NOT removed."""
        recent_ts = self._recent_iso_timestamp(seconds=30)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"created {recent_ts}".encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=inspect_proc) as mock_exec:
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is False
        assert mock_exec.call_count == 1  # inspect only, no rm

    @pytest.mark.asyncio
    async def test_removes_dead_container(self) -> None:
        """A container in 'dead' state older than threshold is removed."""
        old_ts = self._old_iso_timestamp(minutes=10)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"dead {old_ts}".encode(), b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[inspect_proc, rm_proc]):
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is True

    @pytest.mark.asyncio
    async def test_no_conflict_ids_returns_false(self) -> None:
        """When error message has no parseable container IDs, returns False."""
        result = await _remove_conflicting_containers("some unrelated error")
        assert result is False

    @pytest.mark.asyncio
    async def test_inspect_failure_skips_container(self) -> None:
        """If docker inspect fails (container gone), it's skipped."""
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 1
        inspect_proc.communicate = AsyncMock(return_value=(b"", b"No such container"))

        with patch("asyncio.create_subprocess_exec", return_value=inspect_proc):
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is False

    @pytest.mark.asyncio
    async def test_unparseable_timestamp_treated_as_stale(self) -> None:
        """Container with unparseable Created timestamp is treated as stale."""
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(b"created GARBAGE", b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[inspect_proc, rm_proc]):
            result = await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert result is True

    @pytest.mark.asyncio
    async def test_custom_min_age(self) -> None:
        """The min_age_seconds parameter controls the threshold."""
        # 2 minutes old - default 180s threshold should skip
        ts = self._old_iso_timestamp(minutes=2)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"created {ts}".encode(), b""))

        with patch("asyncio.create_subprocess_exec", return_value=inspect_proc):
            result = await _remove_conflicting_containers(self._CONFLICT_MSG, min_age_seconds=180)

        assert result is False

        # Same container with 60s threshold should now be removed
        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[inspect_proc, rm_proc]):
            result = await _remove_conflicting_containers(self._CONFLICT_MSG, min_age_seconds=60)

        assert result is True

    @pytest.mark.asyncio
    async def test_logs_warning_on_removal(self, caplog: pytest.LogCaptureFixture) -> None:
        """A warning is logged when stale conflicting containers are removed."""
        old_ts = self._old_iso_timestamp(minutes=10)
        inspect_proc = AsyncMock()
        inspect_proc.returncode = 0
        inspect_proc.communicate = AsyncMock(return_value=(f"created {old_ts}".encode(), b""))

        rm_proc = AsyncMock()
        rm_proc.returncode = 0
        rm_proc.communicate = AsyncMock(return_value=(b"", b""))

        with patch("asyncio.create_subprocess_exec", side_effect=[inspect_proc, rm_proc]):
            with caplog.at_level(logging.WARNING):
                await _remove_conflicting_containers(self._CONFLICT_MSG)

        assert any("conflicting" in r.message.lower() for r in caplog.records)


class TestSampleInitConflictRetry:
    """sample_init retries on container name conflicts with backoff."""

    _CONFLICT_ERROR = RuntimeError(
        'No services started.\n'
        'Compose up stderr:  Container inspect-excytin-i6gkber-default-1  Creating\n'
        'Error response from daemon: Conflict. The container name '
        '"/inspect-excytin-i6gkber-default-1" is already in use by '
        'container "f782a09e458b". You have to remove (or rename) that '
        'container to be able to reuse that name.'
    )

    @pytest.mark.asyncio
    async def test_retries_on_conflict_and_succeeds(self) -> None:
        """sample_init retries after conflict, even if container already gone."""
        call_count = 0
        sentinel = AsyncMock()

        async def mock_sample_init(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise self._CONFLICT_ERROR
            return {"default": sentinel}

        with (
            patch.object(
                DockerSandboxEnvironment,
                "sample_init",
                side_effect=mock_sample_init,
            ),
            patch(
                "saber.sandbox._remove_conflicting_containers",
                new_callable=AsyncMock,
                return_value=False,  # container already gone (inspect_ai cleaned it)
            ) as mock_remove,
            patch("saber.sandbox.asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            result = await SaberSandboxEnvironment.sample_init("test", None, {})

        assert result == {"default": sentinel}
        mock_remove.assert_called_once_with(str(self._CONFLICT_ERROR), min_age_seconds=0)
        assert call_count == 2
        # Verify backoff sleep was called
        mock_sleep.assert_called_once()
        delay = mock_sleep.call_args[0][0]
        assert delay > 0

    @pytest.mark.asyncio
    async def test_raises_if_not_conflict_error(self) -> None:
        """Non-conflict RuntimeErrors are raised immediately without retry."""
        with patch.object(
            DockerSandboxEnvironment,
            "sample_init",
            side_effect=RuntimeError("something else"),
        ):
            with pytest.raises(RuntimeError, match="something else"):
                await SaberSandboxEnvironment.sample_init("test", None, {})

    @pytest.mark.asyncio
    async def test_retries_even_when_removal_returns_false(self) -> None:
        """Retry happens even when _remove_conflicting_containers returns False.

        inspect_ai's project_cleanup may already remove the container before
        our code sees the error.  Since each retry generates a fresh project
        name, the retry should still succeed.
        """
        call_count = 0
        sentinel = AsyncMock()

        async def mock_sample_init(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise self._CONFLICT_ERROR
            return {"default": sentinel}

        with (
            patch.object(
                DockerSandboxEnvironment,
                "sample_init",
                side_effect=mock_sample_init,
            ),
            patch(
                "saber.sandbox._remove_conflicting_containers",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("saber.sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            result = await SaberSandboxEnvironment.sample_init("test", None, {})

        assert result == {"default": sentinel}
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_success(self) -> None:
        """When sample_init succeeds first try, no removal is attempted."""
        with (
            patch.object(
                DockerSandboxEnvironment,
                "sample_init",
                new_callable=AsyncMock,
                return_value={"default": AsyncMock()},
            ) as mock_super,
            patch(
                "saber.sandbox._remove_conflicting_containers",
                new_callable=AsyncMock,
            ) as mock_remove,
        ):
            await SaberSandboxEnvironment.sample_init("test", None, {})

        mock_super.assert_called_once()
        mock_remove.assert_not_called()

    @pytest.mark.asyncio
    async def test_respects_max_retries(self) -> None:
        """After max retries, the conflict error is raised."""
        with (
            patch.object(
                DockerSandboxEnvironment,
                "sample_init",
                side_effect=self._CONFLICT_ERROR,
            ),
            patch(
                "saber.sandbox._remove_conflicting_containers",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("saber.sandbox.asyncio.sleep", new_callable=AsyncMock),
        ):
            with pytest.raises(RuntimeError, match="already in use"):
                await SaberSandboxEnvironment.sample_init("test", None, {})

    @pytest.mark.asyncio
    async def test_backoff_increases_with_attempts(self) -> None:
        """Each retry waits longer than the previous one."""
        call_count = 0
        sentinel = AsyncMock()

        async def mock_sample_init(*args: object, **kwargs: object) -> dict[str, object]:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                raise self._CONFLICT_ERROR
            return {"default": sentinel}

        sleep_delays: list[float] = []

        async def capture_sleep(delay: float) -> None:
            sleep_delays.append(delay)

        with (
            patch.object(
                DockerSandboxEnvironment,
                "sample_init",
                side_effect=mock_sample_init,
            ),
            patch(
                "saber.sandbox._remove_conflicting_containers",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("saber.sandbox.asyncio.sleep", side_effect=capture_sleep),
            patch("saber.sandbox.random.uniform", return_value=0.0),  # remove jitter for determinism
        ):
            await SaberSandboxEnvironment.sample_init("test", None, {})

        assert call_count == 4  # 3 failures + 1 success
        assert len(sleep_delays) == 3
        # Delays should be monotonically increasing (exponential backoff)
        for i in range(1, len(sleep_delays)):
            assert sleep_delays[i] > sleep_delays[i - 1]
