"""Tests for setup hook auto-discovery.

Tests the ``_discover_setup_hooks()`` function that will be merged into
``saber.task`` in the core package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Load setup_discovery from the local source tree rather than the installed
# ``saber`` package (which doesn't contain this module yet).
_setup_discovery_path = (
    Path(__file__).resolve().parents[1] / "src" / "saber" / "setup_discovery.py"
)
_spec = importlib.util.spec_from_file_location(
    "saber.setup_discovery", _setup_discovery_path
)
assert _spec is not None and _spec.loader is not None
_mod = importlib.util.module_from_spec(_spec)
sys.modules["saber.setup_discovery"] = _mod
_spec.loader.exec_module(_mod)

from saber.setup_discovery import (  # noqa: E402
    _discover_setup_hooks,
    _discover_task_filter,
)


def _clean_fake_modules(domain_name: str) -> None:
    """Remove cached fake domain modules from sys.modules."""
    to_remove = [k for k in sys.modules if k == domain_name or k.startswith(f"{domain_name}.")]
    for k in to_remove:
        del sys.modules[k]


class TestDiscoverSetupHooks:
    """Tests for _discover_setup_hooks()."""

    def test_no_setup_py_returns_empty(self, tmp_path: Path) -> None:
        """Domain without setup.py returns no hooks."""
        result = _discover_setup_hooks(tmp_path, {})
        assert result == []

    def test_setup_py_without_get_hooks_returns_empty(
        self, tmp_path: Path
    ) -> None:
        """setup.py without get_hooks function returns no hooks."""
        domain = tmp_path / "domain_no_hooks"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "# No get_hooks here\ndef other_func(): pass\n"
        )

        try:
            result = _discover_setup_hooks(domain, {})
            assert result == []
        finally:
            _clean_fake_modules("domain_no_hooks")

    def test_setup_py_with_get_hooks_returns_hooks(
        self, tmp_path: Path
    ) -> None:
        """setup.py with valid get_hooks returns the hook list."""
        domain = tmp_path / "domain_with_hooks"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "from saber.hooks import SimpleSetupHook\n"
            "\n"
            "def get_hooks(domain_root: Path) -> list:\n"
            "    return [\n"
            "        SimpleSetupHook(\n"
            '            name="test_hook",\n'
            "            run_fn=lambda dr: None,\n"
            "            guard=lambda dr: True,\n"
            "        )\n"
            "    ]\n"
        )

        try:
            result = _discover_setup_hooks(domain, {})
            assert len(result) == 1
            assert result[0].name == "test_hook"
        finally:
            _clean_fake_modules("domain_with_hooks")

    def test_setup_py_with_empty_get_hooks(self, tmp_path: Path) -> None:
        """get_hooks returning [] yields an empty list."""
        domain = tmp_path / "domain_empty_hooks"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "def get_hooks(domain_root: Path) -> list:\n"
            "    return []\n"
        )

        try:
            result = _discover_setup_hooks(domain, {})
            assert result == []
        finally:
            _clean_fake_modules("domain_empty_hooks")

    def test_run_setup_hooks_succeeds(
        self, tmp_path: Path
    ) -> None:
        """run_setup_hooks() executes hooks and reports success."""
        from saber.hooks import SimpleSetupHook, run_setup_hooks

        sentinel = tmp_path / "hook_ran.txt"
        hook = SimpleSetupHook(
            name="write_sentinel",
            run_fn=lambda dr: sentinel.write_text("ok"),
        )

        hooks_result = run_setup_hooks([hook], tmp_path)

        assert hooks_result.all_succeeded
        assert sentinel.read_text() == "ok"

    def test_run_setup_hooks_failure_produces_error(
        self, tmp_path: Path
    ) -> None:
        """Failed hooks produce a result suitable for RuntimeError."""
        from saber.hooks import SimpleSetupHook, run_setup_hooks

        def _fail(domain_root: Path) -> None:
            msg = "Download failed"
            raise ConnectionError(msg)

        hook = SimpleSetupHook(name="bad_hook", run_fn=_fail)

        hooks_result = run_setup_hooks([hook], tmp_path)
        assert not hooks_result.all_succeeded

        # Simulate the create_task() error handling pattern
        failed = "; ".join(
            f"{r.name}: {r.message}" for r in hooks_result.failed_hooks
        )
        with pytest.raises(RuntimeError, match="bad_hook"):
            raise RuntimeError(f"Setup hook(s) failed: {failed}")

    def test_discover_passes_domain_root_to_get_hooks(
        self, tmp_path: Path
    ) -> None:
        """get_hooks receives the domain_root argument."""
        domain = tmp_path / "domain_root_check"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        # Write a setup.py that captures the received argument
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "received_root = None\n"
            "\n"
            "def get_hooks(domain_root: Path) -> list:\n"
            "    global received_root\n"
            "    received_root = domain_root\n"
            "    return []\n"
        )

        try:
            _discover_setup_hooks(domain, {})

            # Import the module to check what was received
            from saber.task import _import_domain_module

            mod = _import_domain_module(domain, "setup")
            assert mod.received_root == domain  # type: ignore[attr-defined]
        finally:
            _clean_fake_modules("domain_root_check")

    def test_discover_forwards_kwargs_to_get_hooks(
        self, tmp_path: Path
    ) -> None:
        """get_hooks receives named kwargs forwarded from _discover_setup_hooks."""
        domain = tmp_path / "domain_kwargs_fwd"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "captured_kwargs: dict = {}\n"
            "\n"
            "def get_hooks(domain_root: Path, *, benchmarks: str | None = None, force_download: str | None = None) -> list:\n"
            "    captured_kwargs['benchmarks'] = benchmarks\n"
            "    captured_kwargs['force_download'] = force_download\n"
            "    return []\n"
        )

        try:
            extra = {"benchmarks": "a,b", "force_download": "true"}
            _discover_setup_hooks(domain, extra)

            from saber.task import _import_domain_module

            mod = _import_domain_module(domain, "setup")
            assert mod.captured_kwargs["benchmarks"] == "a,b"  # type: ignore[attr-defined]
            assert mod.captured_kwargs["force_download"] == "true"  # type: ignore[attr-defined]
            # Consumed keys should be popped from the dict
            assert "benchmarks" not in extra
            assert "force_download" not in extra
        finally:
            _clean_fake_modules("domain_kwargs_fwd")

    def test_discover_works_with_old_style_get_hooks(
        self, tmp_path: Path
    ) -> None:
        """Old-style get_hooks(domain_root) still works when kwargs are passed."""
        domain = tmp_path / "domain_old_style"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "def get_hooks(domain_root: Path) -> list:\n"
            "    return []\n"
        )

        try:
            # Should not raise, even though extra kwargs are passed
            extra = {"some_extra": "value"}
            result = _discover_setup_hooks(domain, extra)
            assert result == []
            # Unknown kwargs should NOT be popped (not consumed by get_hooks)
            assert "some_extra" in extra
        finally:
            _clean_fake_modules("domain_old_style")


class TestCreateTaskSetupHookIntegration:
    """Tests that create_task() calls the setup hook discovery + execution."""

    def test_create_task_calls_setup_hooks(
        self, tmp_path: Path
    ) -> None:
        """create_task() discovers and runs setup hooks before config loading."""
        from unittest.mock import MagicMock, patch

        from saber.hooks import SetupHooksResult

        mock_hooks = [MagicMock()]
        mock_result = SetupHooksResult(results=())

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                return_value=mock_hooks,
            ) as mock_discover,
            patch(
                "saber.hooks.run_setup_hooks", return_value=mock_result
            ) as mock_run,
            patch("saber.task._find_config_root") as mock_find_config,
        ):
            # _find_config_root is the next call after hooks; stop there
            mock_find_config.side_effect = SystemExit("stop early")

            with pytest.raises(SystemExit, match="stop early"):
                from saber.task import create_task

                create_task(domain_root=tmp_path)

            mock_discover.assert_called_once()
            assert mock_discover.call_args[0][0] == tmp_path
            mock_run.assert_called_once_with(mock_hooks, tmp_path)

    def test_create_task_forwards_dataset_to_setup_hooks(
        self, tmp_path: Path
    ) -> None:
        """create_task() forwards the dataset param to _discover_setup_hooks."""
        from unittest.mock import MagicMock, patch

        from saber.hooks import SetupHooksResult

        mock_hooks = [MagicMock()]
        mock_result = SetupHooksResult(results=())

        # Capture kwargs snapshot at call time (before pop removes dataset)
        captured_kwargs: dict[str, object] = {}

        def capture_discover(domain_root: object, kwargs: dict[str, object]) -> list[object]:
            captured_kwargs.update(kwargs)
            return mock_hooks

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                side_effect=capture_discover,
            ),
            patch(
                "saber.hooks.run_setup_hooks", return_value=mock_result
            ),
            patch("saber.task._find_config_root") as mock_find_config,
        ):
            mock_find_config.side_effect = SystemExit("stop early")

            with pytest.raises(SystemExit, match="stop early"):
                from saber.task import create_task

                create_task(domain_root=tmp_path, dataset="my_dataset")

            # dataset should have been present in the dict at call time
            assert captured_kwargs["dataset"] == "my_dataset"

    def test_create_task_forwards_task_filter_to_setup_hooks(
        self, tmp_path: Path
    ) -> None:
        """create_task() forwards explicit task_filter to _discover_setup_hooks."""
        from unittest.mock import MagicMock, patch

        from saber.hooks import SetupHooksResult

        mock_hooks = [MagicMock()]
        mock_result = SetupHooksResult(results=())
        captured_kwargs: dict[str, object] = {}

        def capture_discover(domain_root: object, kwargs: dict[str, object]) -> list[object]:
            captured_kwargs.update(kwargs)
            return mock_hooks

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                side_effect=capture_discover,
            ),
            patch(
                "saber.hooks.run_setup_hooks", return_value=mock_result
            ),
            patch("saber.task._find_config_root") as mock_find_config,
        ):
            mock_find_config.side_effect = SystemExit("stop early")

            with pytest.raises(SystemExit, match="stop early"):
                from saber.task import create_task

                create_task(
                    domain_root=tmp_path,
                    task_filter="blob_storage_attack_bundle_reconnaissance",
                )

            assert (
                captured_kwargs["task_filter"]
                == "blob_storage_attack_bundle_reconnaissance"
            )

    def test_create_task_omits_dataset_from_hooks_when_none(
        self, tmp_path: Path
    ) -> None:
        """When dataset is None, it is not injected into hook kwargs."""
        from unittest.mock import patch

        from saber.hooks import SetupHooksResult

        mock_result = SetupHooksResult(results=())

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                return_value=[],
            ) as mock_discover,
            patch(
                "saber.hooks.run_setup_hooks", return_value=mock_result
            ),
            patch("saber.task._find_config_root") as mock_find_config,
        ):
            mock_find_config.side_effect = SystemExit("stop early")

            with pytest.raises(SystemExit, match="stop early"):
                from saber.task import create_task

                create_task(domain_root=tmp_path)

            call_args, _ = mock_discover.call_args
            setup_kwargs_dict = call_args[1]
            assert "dataset" not in setup_kwargs_dict

    def test_create_task_raises_on_hook_failure(
        self, tmp_path: Path
    ) -> None:
        """create_task() raises RuntimeError when a setup hook fails."""
        from unittest.mock import patch

        from saber.hooks import (
            SetupHookResult,
            SetupHooksResult,
            SetupHookStatus,
        )

        failed_result = SetupHooksResult(
            results=(
                SetupHookResult(
                    name="bad_download",
                    status=SetupHookStatus.FAILED,
                    message="Connection refused",
                ),
            )
        )

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                return_value=[object()],  # non-empty list triggers run
            ),
            patch(
                "saber.hooks.run_setup_hooks", return_value=failed_result
            ),
        ):
            with pytest.raises(RuntimeError, match="bad_download.*Connection refused"):
                from saber.task import create_task

                create_task(domain_root=tmp_path)

    def test_create_task_skips_run_when_no_hooks(
        self, tmp_path: Path
    ) -> None:
        """create_task() skips run_setup_hooks when discovery returns []."""
        from unittest.mock import patch

        with (
            patch(
                "saber.setup_discovery._discover_setup_hooks",
                return_value=[],
            ),
            patch("saber.hooks.run_setup_hooks") as mock_run,
            patch("saber.task._find_config_root") as mock_find_config,
        ):
            mock_find_config.side_effect = SystemExit("stop early")

            with pytest.raises(SystemExit, match="stop early"):
                from saber.task import create_task

                create_task(domain_root=tmp_path)

            mock_run.assert_not_called()


class TestDiscoverTaskFilter:
    """Tests for _discover_task_filter()."""

    def test_no_setup_py_returns_none(self, tmp_path: Path) -> None:
        """Domain without setup.py returns None."""
        result = _discover_task_filter(tmp_path, "lite")
        assert result is None

    def test_setup_py_without_get_task_filter_returns_none(
        self, tmp_path: Path
    ) -> None:
        """setup.py without get_task_filter function returns None."""
        domain = tmp_path / "domain_no_filter"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "# No get_task_filter here\ndef other_func(): pass\n"
        )

        try:
            result = _discover_task_filter(domain, "lite")
            assert result is None
        finally:
            _clean_fake_modules("domain_no_filter")

    def test_get_task_filter_returns_none_for_unknown_dataset(
        self, tmp_path: Path
    ) -> None:
        """get_task_filter returning None is propagated."""
        domain = tmp_path / "domain_filter_none"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "def get_task_filter(dataset: str) -> str | None:\n"
            "    if dataset == 'lite':\n"
            "        return 'task_a,task_b'\n"
            "    return None\n"
        )

        try:
            result = _discover_task_filter(domain, "unknown")
            assert result is None
        finally:
            _clean_fake_modules("domain_filter_none")

    def test_get_task_filter_returns_filter_string(
        self, tmp_path: Path
    ) -> None:
        """get_task_filter returning a string is propagated."""
        domain = tmp_path / "domain_filter_ok"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "def get_task_filter(dataset: str) -> str | None:\n"
            "    if dataset == 'lite':\n"
            "        return 'task_a,task_b'\n"
            "    return None\n"
        )

        try:
            result = _discover_task_filter(domain, "lite")
            assert result == "task_a,task_b"
        finally:
            _clean_fake_modules("domain_filter_ok")
