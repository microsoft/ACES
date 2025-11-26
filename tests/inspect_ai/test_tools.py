"""Tests for SABER ToolSource implementation."""

import pytest
from unittest.mock import AsyncMock, Mock, patch

from saber.inspect_ai.integration.tools import SABERToolSource, saber_tools


class TestSABERToolSource:
    """Test cases for SABERToolSource class."""

    @pytest.mark.asyncio
    async def test_tools_retrieves_mcp_client_from_sandbox(self):
        """Test that tools() successfully retrieves MCP client from sandbox."""
        # Create mock MCP client with tools
        mock_tool1 = Mock(name="bash")
        mock_tool2 = Mock(name="ls")
        mock_mcp_client = Mock()
        mock_mcp_client.tools = AsyncMock(return_value=[mock_tool1, mock_tool2])

        # Create mock SABER sandbox
        from saber.inspect_ai.saber import SABERSandboxEnvironment
        mock_sandbox = Mock(spec=SABERSandboxEnvironment)
        mock_sandbox._mcp_client = mock_mcp_client
        mock_sandbox._session_id = "test_session"
        mock_sandbox._episode_id = "test_episode"
        mock_sandbox._task_id = "test_task"

        # Mock the sandbox() function to return our mock
        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_sandbox):
            tool_source = SABERToolSource(sandbox_name="saber")
            tools = await tool_source.tools()

            assert len(tools) == 2
            assert tools[0] == mock_tool1
            assert tools[1] == mock_tool2
            mock_mcp_client.tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_unwraps_sandbox_proxy(self):
        """Test that tools() unwraps SandboxEnvironmentProxy to get actual sandbox."""
        # Create mock MCP client
        mock_mcp_client = AsyncMock()
        mock_mcp_client.tools = AsyncMock(return_value=[Mock(name="test_tool")])

        # Create mock SABER sandbox
        mock_actual_sandbox = Mock()
        mock_actual_sandbox._mcp_client = mock_mcp_client
        mock_actual_sandbox._session_id = "test_session"
        mock_actual_sandbox._episode_id = "test_episode"
        mock_actual_sandbox._task_id = "test_task"

        # Create mock proxy that wraps the actual sandbox
        mock_proxy = Mock()
        mock_proxy._sandbox = mock_actual_sandbox

        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_proxy):
            # Mock isinstance to return True for the actual sandbox
            def mock_isinstance(obj, cls):
                return obj == mock_actual_sandbox

            with patch("saber.inspect_ai.integration.tools.isinstance", side_effect=mock_isinstance):
                tool_source = SABERToolSource(sandbox_name="saber")
                tools = await tool_source.tools()

                assert len(tools) == 1
                mock_mcp_client.tools.assert_called_once()

    @pytest.mark.asyncio
    async def test_tools_raises_type_error_for_non_saber_sandbox(self):
        """Test that tools() raises TypeError when sandbox is not a SABERSandboxEnvironment."""
        # Create mock non-SABER sandbox
        mock_sandbox = Mock()

        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_sandbox):
            with patch("saber.inspect_ai.integration.tools.isinstance", return_value=False):
                tool_source = SABERToolSource(sandbox_name="saber")

                with pytest.raises(
                    TypeError, match="Expected SABERSandboxEnvironment, got"
                ):
                    await tool_source.tools()

    @pytest.mark.asyncio
    async def test_tools_raises_runtime_error_when_mcp_client_none(self):
        """Test that tools() raises RuntimeError when MCP client is not available."""
        # Create mock SABER sandbox with no MCP client
        from saber.inspect_ai.saber import SABERSandboxEnvironment
        mock_sandbox = Mock(spec=SABERSandboxEnvironment)
        mock_sandbox._mcp_client = None

        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_sandbox):
            tool_source = SABERToolSource(sandbox_name="saber")

            with pytest.raises(
                RuntimeError, match="SABER MCP client not available"
            ):
                await tool_source.tools()

    @pytest.mark.asyncio
    async def test_tools_propagates_sandbox_retrieval_error(self):
        """Test that tools() propagates errors from sandbox() retrieval."""
        with patch(
            "saber.inspect_ai.integration.tools.sandbox",
            side_effect=ProcessLookupError("No sandbox available"),
        ):
            tool_source = SABERToolSource(sandbox_name="saber")

            with pytest.raises(ProcessLookupError, match="No sandbox available"):
                await tool_source.tools()

    @pytest.mark.asyncio
    async def test_tools_handles_mcp_client_tools_error(self):
        """Test that tools() handles errors from MCP client .tools() call."""
        from saber.inspect_ai.saber import SABERSandboxEnvironment
        mock_mcp_client = Mock()
        mock_mcp_client.tools = AsyncMock(
            side_effect=Exception("MCP server error")
        )

        mock_sandbox = Mock(spec=SABERSandboxEnvironment)
        mock_sandbox._mcp_client = mock_mcp_client
        mock_sandbox._session_id = "test_session"
        mock_sandbox._episode_id = "test_episode"
        mock_sandbox._task_id = "test_task"

        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_sandbox):
            tool_source = SABERToolSource(sandbox_name="saber")

            with pytest.raises(Exception, match="MCP server error"):
                await tool_source.tools()

    @pytest.mark.asyncio
    async def test_tools_with_custom_sandbox_name(self):
        """Test that tools() works with custom sandbox name."""
        from saber.inspect_ai.saber import SABERSandboxEnvironment
        mock_mcp_client = Mock()
        mock_mcp_client.tools = AsyncMock(return_value=[Mock(name="custom_tool")])

        mock_sandbox = Mock(spec=SABERSandboxEnvironment)
        mock_sandbox._mcp_client = mock_mcp_client
        mock_sandbox._session_id = "test_session"
        mock_sandbox._episode_id = "test_episode"
        mock_sandbox._task_id = "test_task"

        with patch("saber.inspect_ai.integration.tools.sandbox", return_value=mock_sandbox) as mock_sb:
            tool_source = SABERToolSource(sandbox_name="custom_saber")
            tools = await tool_source.tools()

            # Verify sandbox was called with custom name
            mock_sb.assert_called_once_with("custom_saber")
            assert len(tools) == 1


class TestSaberToolsFactory:
    """Test cases for saber_tools() factory function."""

    def test_saber_tools_returns_tool_source_instance(self):
        """Test that saber_tools() returns SABERToolSource instance."""
        tool_source = saber_tools()
        assert isinstance(tool_source, SABERToolSource)

    def test_saber_tools_with_custom_sandbox_name(self):
        """Test that saber_tools() accepts custom sandbox name."""
        tool_source = saber_tools(sandbox_name="my_sandbox")
        assert isinstance(tool_source, SABERToolSource)
        assert tool_source._sandbox_name == "my_sandbox"

    def test_saber_tools_default_sandbox_name(self):
        """Test that saber_tools() uses default sandbox name."""
        tool_source = saber_tools()
        assert tool_source._sandbox_name == "saber"
