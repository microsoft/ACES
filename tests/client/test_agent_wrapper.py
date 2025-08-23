"""
Tests for AgentWrapper and AgentLoader functionality.

Tests the generic agent adaptation system that allows any agent
to work with SABER without code changes.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest

from saber.client.agent_wrapper import AgentLoader, AgentWrapper


# Test agent classes for testing
class SyncTestAgent:
    """Test agent with sync process method."""

    def __init__(self):
        self.reset_called = False
        self.process_calls = []

    def process(self, prompt):
        self.process_calls.append(prompt)
        return f"response to: {prompt}"

    def reset(self):
        self.reset_called = True
        self.process_calls = []


class AsyncTestAgent:
    """Test agent with async respond method."""

    def __init__(self):
        self.clear_called = False
        self.respond_calls = []

    async def respond(self, message):
        self.respond_calls.append(message)
        return f"async response to: {message}"

    async def clear(self):
        self.clear_called = True
        self.respond_calls = []


class CustomMethodAgent:
    """Test agent with custom method names."""

    def __init__(self):
        self.calls = []

    def generate(self, input_text):
        self.calls.append(input_text)
        return f"generated: {input_text}"

    def initialize(self):
        self.calls = []


class CallableAgent:
    """Test agent that is itself callable."""

    def __init__(self):
        self.calls = []

    def __call__(self, prompt):
        self.calls.append(prompt)
        return f"callable response: {prompt}"


def simple_function_agent(query):
    """Test function-based agent."""
    return f"function response: {query}"


class MultipleMethodAgent:
    """Test agent with multiple possible methods."""

    def process(self, prompt):
        return f"process: {prompt}"

    def generate(self, prompt):
        return f"generate: {prompt}"

    def respond(self, prompt):
        return f"respond: {prompt}"


class AgentWithoutValidMethods:
    """Test agent without valid processing methods."""

    def some_other_method(self):
        return "not a valid agent method"


@pytest.fixture
def sync_agent():
    """Sync test agent fixture."""
    return SyncTestAgent()


@pytest.fixture
def async_agent():
    """Async test agent fixture."""
    return AsyncTestAgent()


@pytest.fixture
def custom_agent():
    """Custom method name agent fixture."""
    return CustomMethodAgent()


@pytest.fixture
def callable_agent():
    """Callable agent fixture."""
    return CallableAgent()


class TestAgentWrapper:
    """Test cases for AgentWrapper."""

    def test_wrap_custom_method_agent(self, custom_agent):
        """Test wrapping agent with custom method names."""
        wrapper = AgentWrapper(custom_agent)

        assert wrapper.process_method == custom_agent.generate
        assert wrapper.reset_method == custom_agent.initialize
        assert not wrapper.is_async

    def test_wrap_callable_agent(self, callable_agent):
        """Test wrapping callable agent."""
        wrapper = AgentWrapper(callable_agent)

        assert wrapper.process_method == callable_agent.__call__
        assert wrapper.reset_method is None  # No reset method
        assert not wrapper.is_async

    def test_wrap_function_agent(self):
        """Test wrapping function-based agent."""
        wrapper = AgentWrapper(simple_function_agent)

        assert wrapper.agent == simple_function_agent
        assert callable(wrapper.process_method)
        # For function agents, the process_method is the __call__ method
        assert hasattr(wrapper.process_method, "__call__")
        assert wrapper.reset_method is None
        assert not wrapper.is_async

    def test_wrap_invalid_agent(self):
        """Test wrapping agent without valid methods."""
        agent = AgentWithoutValidMethods()

        with pytest.raises(ValueError, match="Could not detect processing method"):
            AgentWrapper(agent)

    @pytest.mark.asyncio
    async def test_process_prompt_sync(self, sync_agent):
        """Test process_prompt with sync agent."""
        wrapper = AgentWrapper(sync_agent)

        result = await wrapper.process_prompt("test prompt")

        assert result == "response to: test prompt"
        assert sync_agent.process_calls == ["test prompt"]

    @pytest.mark.asyncio
    async def test_process_prompt_async(self, async_agent):
        """Test process_prompt with async agent."""
        wrapper = AgentWrapper(async_agent)

        result = await wrapper.process_prompt("test message")

        assert result == "async response to: test message"
        assert async_agent.respond_calls == ["test message"]

    @pytest.mark.asyncio
    async def test_process_prompt_custom_param(self, custom_agent):
        """Test process_prompt with custom parameter name."""
        wrapper = AgentWrapper(custom_agent)

        result = await wrapper.process_prompt("test input")

        assert result == "generated: test input"
        assert custom_agent.calls == ["test input"]

    @pytest.mark.asyncio
    async def test_process_prompt_non_string_result(self):
        """Test process_prompt with non-string result."""

        class NumericAgent:
            def process(self, prompt):
                return 42

        wrapper = AgentWrapper(NumericAgent())
        result = await wrapper.process_prompt("test")

        assert result == "42"

    @pytest.mark.asyncio
    async def test_reset_sync(self, sync_agent):
        """Test reset with sync agent."""
        wrapper = AgentWrapper(sync_agent)

        await wrapper.reset()

        assert sync_agent.reset_called

    @pytest.mark.asyncio
    async def test_reset_async(self, async_agent):
        """Test reset with async agent."""
        wrapper = AgentWrapper(async_agent)

        await wrapper.reset()

        assert async_agent.clear_called

    @pytest.mark.asyncio
    async def test_reset_no_method(self, callable_agent):
        """Test reset when agent has no reset method."""
        wrapper = AgentWrapper(callable_agent)

        # Should not raise error
        await wrapper.reset()


class TestAgentLoader:
    """Test cases for AgentLoader."""

    def test_load_from_path_simple_class(self):
        """Test loading agent from file with simple class."""
        # Create temporary agent file
        agent_code = """
class TestAgent:
    def process(self, prompt):
        return f"loaded: {prompt}"

    def reset(self):
        pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                wrapper = AgentLoader.load_from_path(f.name)

                assert isinstance(wrapper, AgentWrapper)
                assert wrapper.agent.__class__.__name__ == "TestAgent"
                assert wrapper.process_method.__name__ == "process"
            finally:
                Path(f.name).unlink()

    def test_load_from_path_specific_class(self):
        """Test loading specific class from file."""
        agent_code = """
class FirstAgent:
    def process(self, prompt):
        return "first"

class SecondAgent:
    def generate(self, prompt):
        return "second"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                wrapper = AgentLoader.load_from_path(f.name, "SecondAgent")

                assert wrapper.agent.__class__.__name__ == "SecondAgent"
                assert wrapper.process_method.__name__ == "generate"
            finally:
                Path(f.name).unlink()

    def test_load_from_path_function(self):
        """Test loading function-based agent from file."""
        agent_code = """
def my_agent_function(prompt):
    return f"function result: {prompt}"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                wrapper = AgentLoader.load_from_path(f.name)

                assert callable(wrapper.agent)
                # For function agents, the process_method is the __call__ method
                assert callable(wrapper.process_method)
            finally:
                Path(f.name).unlink()

    def test_load_from_path_multiple_agents(self):
        """Test loading from file with multiple agent classes."""
        agent_code = """
class SecurityAgent:
    def process(self, prompt):
        return "security"

class DataAgent:
    def analyze(self, prompt):
        return "data"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                wrapper = AgentLoader.load_from_path(f.name)

                # Should prefer class with "agent" in name
                assert wrapper.agent.__class__.__name__ == "SecurityAgent"
            finally:
                Path(f.name).unlink()

    def test_load_from_path_file_not_found(self):
        """Test loading from non-existent file."""
        with pytest.raises(FileNotFoundError):
            AgentLoader.load_from_path("/nonexistent/file.py")

    def test_load_from_path_no_agent_found(self):
        """Test loading from file with no valid agents."""
        agent_code = """
class NotAnAgent:
    def some_method(self):
        return "not an agent"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                with pytest.raises(ValueError, match="No agent found in module"):
                    AgentLoader.load_from_path(f.name)
            finally:
                Path(f.name).unlink()

    def test_load_from_path_class_not_found(self):
        """Test loading specific class that doesn't exist."""
        agent_code = """
class ExistingAgent:
    def process(self, prompt):
        return "exists"
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(agent_code)
            f.flush()

            try:
                with pytest.raises(AttributeError, match="Class 'NonExistentAgent' not found"):
                    AgentLoader.load_from_path(f.name, "NonExistentAgent")
            finally:
                Path(f.name).unlink()
