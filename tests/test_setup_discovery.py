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

from saber.setup_discovery import _discover_setup_hooks  # noqa: E402


def _clean_fake_modules(domain_name: str) -> None:
    """Remove cached fake domain modules from sys.modules."""
    to_remove = [k for k in sys.modules if k == domain_name or k.startswith(f"{domain_name}.")]
    for k in to_remove:
        del sys.modules[k]


class TestDiscoverSetupHooks:
    """Tests for _discover_setup_hooks()."""

    def test_no_setup_py_returns_empty(self, tmp_path: Path) -> None:
        """Domain without setup.py returns no hooks."""
        result = _discover_setup_hooks(tmp_path)
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
            result = _discover_setup_hooks(domain)
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
            result = _discover_setup_hooks(domain)
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
            result = _discover_setup_hooks(domain)
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
            _discover_setup_hooks(domain)

            # Import the module to check what was received
            from saber.task import _import_domain_module

            mod = _import_domain_module(domain, "setup")
            assert mod.received_root == domain  # type: ignore[attr-defined]
        finally:
            _clean_fake_modules("domain_root_check")

    def test_discover_forwards_kwargs_to_get_hooks(
        self, tmp_path: Path
    ) -> None:
        """get_hooks receives **kwargs forwarded from _discover_setup_hooks."""
        domain = tmp_path / "domain_kwargs_fwd"
        domain.mkdir()
        (domain / "__init__.py").write_text("")
        (domain / "setup.py").write_text(
            "from pathlib import Path\n"
            "\n"
            "captured_kwargs: dict = {}\n"
            "\n"
            "def get_hooks(domain_root: Path, **kwargs: object) -> list:\n"
            "    captured_kwargs.update(kwargs)\n"
            "    return []\n"
        )

        try:
            _discover_setup_hooks(
                domain, benchmarks="a,b", force_download="true"
            )

            from saber.task import _import_domain_module

            mod = _import_domain_module(domain, "setup")
            assert mod.captured_kwargs["benchmarks"] == "a,b"  # type: ignore[attr-defined]
            assert mod.captured_kwargs["force_download"] == "true"  # type: ignore[attr-defined]
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
            result = _discover_setup_hooks(domain, some_extra="value")
            assert result == []
        finally:
            _clean_fake_modules("domain_old_style")
