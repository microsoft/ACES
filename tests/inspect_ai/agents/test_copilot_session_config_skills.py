"""Tests for Copilot skills directory support in CopilotSessionConfig.

This module tests the skill_directories feature added to CopilotSessionConfig:
1. to_dict() serialization behavior (include/exclude based on None)
2. create() factory method parameter support

Note: The SDK expects camelCase keys ('skill_directories', 'available_tools')
while the Python dataclass uses snake_case ('skill_directories', 'available_tools').
"""

import pytest
from unittest.mock import MagicMock

from saber.inspect_ai.agents.registry.copilot.models import (
    CopilotSessionConfig,
    SystemMessageConfig,
)


class TestCopilotSessionConfigToDict:
    """Tests for CopilotSessionConfig.to_dict() with skill_directories."""

    def _create_minimal_config(
        self, skill_directories: list[str] | None = None
    ) -> CopilotSessionConfig:
        """Create a minimal CopilotSessionConfig for testing."""
        return CopilotSessionConfig(
            model="gpt-5",
            tools=[],
            available_tools=["report_intent"],
            system_message=SystemMessageConfig(content="Test system message"),
            streaming=False,
            provider=None,
            skill_directories=skill_directories,
        )

    def test_skill_directories_none_not_in_output(self) -> None:
        """When skill_directories=None, the field should NOT be in to_dict() output."""
        config = self._create_minimal_config(skill_directories=None)

        result = config.to_dict()

        # SDK expects camelCase 'skill_directories'
        assert "skill_directories" not in result
        # Verify other expected fields are present
        assert "model" in result
        assert result["model"] == "gpt-5"

    def test_skill_directories_with_paths_in_output(self) -> None:
        """When skill_directories has paths, the field should be in to_dict() output."""
        config = self._create_minimal_config(
            skill_directories=["/path/to/skills", "/another/skills/dir"]
        )

        result = config.to_dict()

        # SDK expects camelCase 'skill_directories'
        assert "skill_directories" in result
        assert result["skill_directories"] == ["/path/to/skills", "/another/skills/dir"]

    def test_skill_directories_single_path_in_output(self) -> None:
        """When skill_directories has a single path, it should be properly included."""
        config = self._create_minimal_config(skill_directories=["/single/path"])

        result = config.to_dict()

        # SDK expects camelCase 'skill_directories'
        assert "skill_directories" in result
        assert result["skill_directories"] == ["/single/path"]

    def test_skill_directories_empty_list_in_output(self) -> None:
        """When skill_directories is an empty list, the field SHOULD be in output.

        An empty list is not None, so it should be included (though unusual).
        """
        config = self._create_minimal_config(skill_directories=[])

        result = config.to_dict()

        # SDK expects camelCase 'skill_directories'
        assert "skill_directories" in result
        assert result["skill_directories"] == []


class TestCopilotSessionConfigCreate:
    """Tests for CopilotSessionConfig.create() factory method with skill_directories."""

    def test_create_without_skill_directories(self) -> None:
        """create() without skill_directories should produce config with None."""
        config = CopilotSessionConfig.create(
            model="gpt-5",
            tools=[],
            system_content="Test system content",
        )

        assert config.skill_directories is None
        # Verify to_dict doesn't include it (SDK expects camelCase)
        assert "skill_directories" not in config.to_dict()

    def test_create_with_skill_directories(self) -> None:
        """create() with skill_directories should set the value correctly."""
        config = CopilotSessionConfig.create(
            model="gpt-5",
            tools=[],
            system_content="Test system content",
            skill_directories=["/path/to/skills"],
        )

        assert config.skill_directories == ["/path/to/skills"]
        # Verify to_dict includes it (SDK expects camelCase)
        result = config.to_dict()
        assert "skill_directories" in result
        assert result["skill_directories"] == ["/path/to/skills"]
        # Verify 'skill' tool is added to available_tools when skill_directories provided
        assert "skill" in result["available_tools"]

    def test_create_with_multiple_skill_directories(self) -> None:
        """create() with multiple skill_directories should preserve all paths."""
        skill_paths = ["/skills/dir1", "/skills/dir2", "/skills/dir3"]

        config = CopilotSessionConfig.create(
            model="gpt-5",
            tools=[],
            system_content="Test system content",
            skill_directories=skill_paths,
        )

        assert config.skill_directories == skill_paths
        # SDK expects camelCase 'skill_directories'
        assert config.to_dict()["skill_directories"] == skill_paths

    def test_create_with_empty_skill_directories(self) -> None:
        """create() with empty list should set empty list (not None)."""
        config = CopilotSessionConfig.create(
            model="gpt-5",
            tools=[],
            system_content="Test system content",
            skill_directories=[],
        )

        assert config.skill_directories == []
        # Empty list should still be in output (SDK expects camelCase)
        assert "skill_directories" in config.to_dict()

    def test_create_preserves_other_parameters_with_skill_directories(self) -> None:
        """create() with skill_directories should not affect other parameters."""
        mock_tool = MagicMock()
        mock_tool.name = "test_tool"

        config = CopilotSessionConfig.create(
            model="claude-sonnet-4",
            tools=[mock_tool],
            system_content="Custom system message",
            streaming=True,
            system_mode="replace",
            skill_directories=["/custom/skills"],
        )

        assert config.model == "claude-sonnet-4"
        assert config.tools == [mock_tool]
        assert config.streaming is True
        assert config.system_message.content == "Custom system message"
        assert config.system_message.mode == "replace"
        assert config.skill_directories == ["/custom/skills"]
        # Check that report_intent is added to available_tools
        assert "report_intent" in config.available_tools
        assert "test_tool" in config.available_tools


class TestSkillsDirParsing:
    """Tests for skills_dir comma-separated parsing logic.

    This tests the parsing logic used in solver_factory.py to convert the
    skills_dir string parameter to the skill_directories list:
        [p.strip() for p in skills_dir.split(",")] if skills_dir else None
    """

    @staticmethod
    def _parse_skills_dir(skills_dir: str | None) -> list[str] | None:
        """Helper that replicates the parsing logic from solver_factory.py."""
        return [p.strip() for p in skills_dir.split(",")] if skills_dir else None

    def test_none_returns_none(self) -> None:
        """None input should produce None output."""
        result = self._parse_skills_dir(None)
        assert result is None

    def test_empty_string_returns_none(self) -> None:
        """Empty string input should produce None output (falsy)."""
        result = self._parse_skills_dir("")
        assert result is None

    def test_single_path(self) -> None:
        """Single path should produce single-element list."""
        result = self._parse_skills_dir("/path/to/skills")
        assert result == ["/path/to/skills"]

    def test_single_path_with_trailing_slash(self) -> None:
        """Single path with trailing slash should be preserved."""
        result = self._parse_skills_dir("/path/to/skills/")
        assert result == ["/path/to/skills/"]

    def test_multiple_paths_comma_separated(self) -> None:
        """Comma-separated paths should produce list."""
        result = self._parse_skills_dir("/path/one,/path/two,/path/three")
        assert result == ["/path/one", "/path/two", "/path/three"]

    def test_two_paths_comma_separated(self) -> None:
        """Two comma-separated paths should produce two-element list."""
        result = self._parse_skills_dir("/path/one,/path/two")
        assert result == ["/path/one", "/path/two"]

    def test_paths_with_leading_spaces_are_stripped(self) -> None:
        """Leading spaces around paths should be stripped."""
        result = self._parse_skills_dir("/path/one, /path/two, /path/three")
        assert result == ["/path/one", "/path/two", "/path/three"]

    def test_paths_with_trailing_spaces_are_stripped(self) -> None:
        """Trailing spaces around paths should be stripped."""
        result = self._parse_skills_dir("/path/one ,/path/two ,/path/three")
        assert result == ["/path/one", "/path/two", "/path/three"]

    def test_paths_with_mixed_spaces_are_stripped(self) -> None:
        """Mixed spaces around paths should be stripped."""
        result = self._parse_skills_dir("/path/one , /path/two , /path/three")
        assert result == ["/path/one", "/path/two", "/path/three"]

    def test_paths_with_no_spaces(self) -> None:
        """Paths with no spaces should work correctly."""
        result = self._parse_skills_dir("/a,/b,/c")
        assert result == ["/a", "/b", "/c"]

    def test_relative_paths(self) -> None:
        """Relative paths should be preserved."""
        result = self._parse_skills_dir("./skills,../other/skills")
        assert result == ["./skills", "../other/skills"]

    def test_windows_style_paths(self) -> None:
        """Windows-style paths should be preserved (no comma in path)."""
        result = self._parse_skills_dir("C:\\Users\\skills,D:\\other\\skills")
        assert result == ["C:\\Users\\skills", "D:\\other\\skills"]
