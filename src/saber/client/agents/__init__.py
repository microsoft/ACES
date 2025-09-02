"""Client agents package.

Contains test and demo agents for SABER benchmarking and UI testing.
"""

from .dummy_test_agent import DummyTestAgent
from .dummy_test_agent import main as dummy_test_agent_main

__all__ = ["DummyTestAgent", "dummy_test_agent_main"]
