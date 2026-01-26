"""Tests for tool name mapping.

Tests cover:
- Mapping known Claude tools to SABER MCP tools
- Handling unknown tools
- Empty tool list handling
- Special cases like Skill tool
"""



class TestClaudeToSaberToolMap:
    """Tests for CLAUDE_TO_SABER_TOOL_MAP constant."""

    def test_map_contains_all_expected_tools(self):
        """Test that the mapping contains all expected Claude tools."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        expected_tools = [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "WebFetch",
            "WebSearch",
            "Grep",
            "Glob",
            "LS",
            "Skill",
        ]

        for tool in expected_tools:
            assert tool in CLAUDE_TO_SABER_TOOL_MAP, f"Missing tool: {tool}"

    def test_bash_maps_correctly(self):
        """Test Bash -> bash mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Bash"] == "bash"

    def test_read_maps_correctly(self):
        """Test Read -> read_file mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Read"] == "read_file"

    def test_write_maps_correctly(self):
        """Test Write -> write_file mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Write"] == "write_file"

    def test_edit_maps_correctly(self):
        """Test Edit -> edit_file mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Edit"] == "edit_file"

    def test_webfetch_maps_correctly(self):
        """Test WebFetch -> fetch_webpage mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["WebFetch"] == "fetch_webpage"

    def test_websearch_maps_correctly(self):
        """Test WebSearch -> web_search mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["WebSearch"] == "web_search"

    def test_grep_maps_correctly(self):
        """Test Grep -> grep mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Grep"] == "grep"

    def test_glob_maps_correctly(self):
        """Test Glob -> glob mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Glob"] == "glob"

    def test_ls_maps_correctly(self):
        """Test LS -> list_dir mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["LS"] == "list_dir"

    def test_skill_maps_to_none(self):
        """Test Skill -> None mapping (handled via skill_directories)."""
        from saber.inspect_ai.agents.registry.custom_agent import CLAUDE_TO_SABER_TOOL_MAP

        assert CLAUDE_TO_SABER_TOOL_MAP["Skill"] is None


class TestMapToolsToSaber:
    """Tests for map_tools_to_saber function."""

    def test_map_single_tool(self):
        """Test mapping a single tool."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        result = map_tools_to_saber(["Bash"])
        assert result == ["bash"]

    def test_map_multiple_tools(self):
        """Test mapping multiple tools."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        claude_tools = ["Bash", "Read", "Write", "Edit"]
        result = map_tools_to_saber(claude_tools)

        assert result == ["bash", "read_file", "write_file", "edit_file"]

    def test_map_all_tools(self):
        """Test mapping all known Claude tools."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        claude_tools = [
            "Bash",
            "Read",
            "Write",
            "Edit",
            "WebFetch",
            "WebSearch",
            "Grep",
            "Glob",
            "LS",
        ]
        result = map_tools_to_saber(claude_tools)

        expected = [
            "bash",
            "read_file",
            "write_file",
            "edit_file",
            "fetch_webpage",
            "web_search",
            "grep",
            "glob",
            "list_dir",
        ]
        assert result == expected

    def test_empty_tool_list(self):
        """Test that empty list returns empty list."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        result = map_tools_to_saber([])
        assert result == []

    def test_skill_tool_is_filtered_out(self):
        """Test that Skill tool is filtered out (not included in result)."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        claude_tools = ["Bash", "Skill", "Read"]
        result = map_tools_to_saber(claude_tools)

        # Skill should be filtered out because it maps to None
        assert result == ["bash", "read_file"]
        assert "Skill" not in result
        assert None not in result

    def test_preserves_order(self):
        """Test that tool order is preserved during mapping."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        claude_tools = ["Read", "Bash", "Write"]
        result = map_tools_to_saber(claude_tools)

        assert result == ["read_file", "bash", "write_file"]

    def test_unknown_tool_kept_as_is(self):
        """Test that unknown tools are kept with their original name."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        # MCP tools like mcp__agent-orchestrator__run_agent should pass through
        claude_tools = [
            "Bash",
            "mcp__agent-orchestrator__run_agent",
            "custom_tool",
        ]
        result = map_tools_to_saber(claude_tools)

        assert result == ["bash", "mcp__agent-orchestrator__run_agent", "custom_tool"]

    def test_already_saber_tool_names_pass_through(self):
        """Test that tools already in SABER format pass through unchanged."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        # If user specifies SABER tool names directly, they should work
        saber_tools = ["bash", "read_file", "write_file"]
        result = map_tools_to_saber(saber_tools)

        # They pass through as-is (not in mapping, so kept original)
        assert result == ["bash", "read_file", "write_file"]

    def test_mixed_claude_and_mcp_tools(self):
        """Test mixing Claude tools with MCP-style tools."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        mixed_tools = [
            "Bash",  # Claude tool
            "Read",  # Claude tool
            "mcp__agent-orchestrator__run_agent",  # MCP tool
            "mcp__agent-orchestrator__get_agent_status",  # MCP tool
        ]
        result = map_tools_to_saber(mixed_tools)

        expected = [
            "bash",
            "read_file",
            "mcp__agent-orchestrator__run_agent",
            "mcp__agent-orchestrator__get_agent_status",
        ]
        assert result == expected


class TestMapToolsToSaberEdgeCases:
    """Edge case tests for map_tools_to_saber."""

    def test_duplicate_tools_preserved(self):
        """Test that duplicate tools are preserved (no deduplication)."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        claude_tools = ["Bash", "Bash", "Read"]
        result = map_tools_to_saber(claude_tools)

        # Duplicates are preserved
        assert result == ["bash", "bash", "read_file"]

    def test_case_sensitive_mapping(self):
        """Test that mapping is case-sensitive."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        # Lowercase "bash" is not "Bash" so should pass through
        claude_tools = ["bash", "BASH", "Bash"]
        result = map_tools_to_saber(claude_tools)

        # Only "Bash" gets mapped, others pass through
        assert result == ["bash", "BASH", "bash"]

    def test_only_skill_tool_returns_empty(self):
        """Test that list with only Skill tool returns empty list."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        result = map_tools_to_saber(["Skill"])
        assert result == []

    def test_multiple_skills_filtered(self):
        """Test that multiple Skill entries are all filtered."""
        from saber.inspect_ai.agents.registry.custom_agent import map_tools_to_saber

        result = map_tools_to_saber(["Skill", "Bash", "Skill", "Read", "Skill"])
        assert result == ["bash", "read_file"]
