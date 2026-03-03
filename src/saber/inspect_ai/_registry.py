"""Registry for SABER Inspect AI extension.

This module is imported by Inspect AI via the setuptools entry point mechanism.
It ensures that the SABER sandbox environment is registered when the saber
package is installed.
"""

from ..sandbox import SaberSandboxEnvironment  # noqa: F401
from .integration.agent_model import agent  # noqa: F401 - registers @modelapi("agent")
from .integration.tools import saber_tools  # noqa: F401

__all__ = ["SaberSandboxEnvironment", "saber_tools", "agent"]
