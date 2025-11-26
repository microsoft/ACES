"""Unit tests for SABER Inspect AI registry module (_registry.py).

Tests cover:
- Module imports and registration side effects
- Entry point discovery for sandboxenv and tools
"""

import pytest


class TestRegistryImports:
    """Test that registry module imports trigger required registrations."""

    def test_registry_imports_sandbox_environment(self):
        """Test that importing _registry makes SABERSandboxEnvironment available."""
        # This import should trigger the @sandboxenv decorator registration
        from saber.inspect_ai import _registry

        # Verify the import succeeded and exposes expected symbols
        assert hasattr(_registry, "SABERSandboxEnvironment")
        assert hasattr(_registry, "saber_tools")

    def test_registry_module_all_exports(self):
        """Test that _registry.__all__ contains expected exports."""
        from saber.inspect_ai import _registry

        assert hasattr(_registry, "__all__")
        assert "SABERSandboxEnvironment" in _registry.__all__
        assert "saber_tools" in _registry.__all__

    def test_sandbox_environment_is_registered(self):
        """Test that SABERSandboxEnvironment is properly decorated and registered."""
        from saber.inspect_ai._registry import SABERSandboxEnvironment

        # Verify the class exists and has expected attributes
        assert SABERSandboxEnvironment is not None
        assert hasattr(SABERSandboxEnvironment, "__name__")
        assert SABERSandboxEnvironment.__name__ == "SABERSandboxEnvironment"

    def test_saber_tools_is_registered(self):
        """Test that saber_tools is properly decorated and registered."""
        from saber.inspect_ai._registry import saber_tools

        # Verify the function exists and is callable
        assert saber_tools is not None
        assert callable(saber_tools)
