"""Registry for SABER Inspect AI extension.

This module is imported by Inspect AI via the setuptools entry point mechanism.
It ensures that the SABER sandbox environment is registered when the saber
package is installed.
"""

from .integration.tools import saber_tools  # noqa: F401

# Import to trigger @sandboxenv decorator registration
from .saber import SABERSandboxEnvironment  # noqa: F401

__all__ = ["SABERSandboxEnvironment", "saber_tools"]
