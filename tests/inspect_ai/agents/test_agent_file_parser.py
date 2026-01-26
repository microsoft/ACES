"""Tests for agent file parser.

Tests cover:
- Parsing a valid agent.md file with YAML frontmatter
- Extracting metadata and prompt content
- Handling missing optional fields
- Error handling for malformed YAML
"""

from pathlib import Path

import pytest


class TestParseAgentFile:
    """Tests for parse_agent_file function."""

    def test_parse_valid_agent_file(self, tmp_path: Path):
        """Test parsing a valid agent.md file with full frontmatter."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Red Team Agent"
description: "Execute security testing tasks"
version: "1.0.0"
model: "claude-sonnet-4-5-20250929"
tools:
  allowed:
    - "Bash"
    - "Read"
    - "Write"
  disallowed: []
permission_mode: "acceptEdits"
max_turns: 100
---

# Red Team Agent

You are a security researcher. Execute tasks directly.

## Instructions

1. Read the task
2. Execute the plan
3. Report results
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, prompt_content = parse_agent_file(agent_file)

        assert metadata.name == "Red Team Agent"
        assert metadata.description == "Execute security testing tasks"
        assert metadata.version == "1.0.0"
        assert metadata.model == "claude-sonnet-4-5-20250929"
        assert metadata.tools is not None
        assert metadata.tools.allowed == ["Bash", "Read", "Write"]
        assert metadata.tools.disallowed == []
        assert metadata.permission_mode == "acceptEdits"
        assert metadata.max_turns == 100

        assert "# Red Team Agent" in prompt_content
        assert "You are a security researcher." in prompt_content
        assert "---" not in prompt_content  # Frontmatter should be stripped

    def test_parse_minimal_agent_file(self, tmp_path: Path):
        """Test parsing agent file with only required fields."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Minimal Agent"
---

# Minimal Agent

Just a simple prompt.
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, prompt_content = parse_agent_file(agent_file)

        assert metadata.name == "Minimal Agent"
        assert metadata.description is None
        assert metadata.version is None
        assert metadata.model is None
        assert metadata.tools is None
        assert metadata.permission_mode is None
        assert metadata.max_turns is None

        assert "# Minimal Agent" in prompt_content
        assert "Just a simple prompt." in prompt_content

    def test_parse_file_with_string_path(self, tmp_path: Path):
        """Test parsing with string path instead of Path object."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "String Path Agent"
---

# Content
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        # Pass as string
        metadata, prompt_content = parse_agent_file(str(agent_file))

        assert metadata.name == "String Path Agent"

    def test_parse_file_with_tools_allowed_only(self, tmp_path: Path):
        """Test parsing with only allowed tools specified."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Limited Tools Agent"
tools:
  allowed:
    - "Read"
    - "Write"
---

# Agent
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, _ = parse_agent_file(agent_file)

        assert metadata.tools is not None
        assert metadata.tools.allowed == ["Read", "Write"]
        assert metadata.tools.disallowed is None

    def test_parse_file_with_tools_disallowed_only(self, tmp_path: Path):
        """Test parsing with only disallowed tools specified."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Restricted Agent"
tools:
  disallowed:
    - "Bash"
    - "WebFetch"
---

# Agent
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, _ = parse_agent_file(agent_file)

        assert metadata.tools is not None
        assert metadata.tools.allowed is None
        assert metadata.tools.disallowed == ["Bash", "WebFetch"]

    def test_parse_file_preserves_multiline_content(self, tmp_path: Path):
        """Test that multiline prompt content is preserved correctly."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Multiline Agent"
---

# Header

First paragraph.

Second paragraph with **bold** and `code`.

## Section

- Item 1
- Item 2

```python
print("code block")
```
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        _, prompt_content = parse_agent_file(agent_file)

        assert "# Header" in prompt_content
        assert "First paragraph." in prompt_content
        assert "Second paragraph with **bold** and `code`." in prompt_content
        assert "## Section" in prompt_content
        assert "- Item 1" in prompt_content
        assert '```python\nprint("code block")\n```' in prompt_content


class TestParseAgentFileErrors:
    """Tests for error handling in parse_agent_file."""

    def test_file_not_found_raises(self):
        """Test that missing file raises FileNotFoundError."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        with pytest.raises(FileNotFoundError):
            parse_agent_file("/nonexistent/path/agent.md")

    def test_no_frontmatter_raises(self, tmp_path: Path):
        """Test that file without frontmatter raises ValueError."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """# Agent Without Frontmatter

Just content, no YAML.
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_unclosed_frontmatter_raises(self, tmp_path: Path):
        """Test that unclosed frontmatter raises ValueError."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Unclosed"

# This frontmatter never closes
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_invalid_yaml_raises(self, tmp_path: Path):
        """Test that malformed YAML raises ValueError."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Test"
invalid_yaml: [unclosed bracket
nested:
  - broken
    indentation
---

# Content
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        with pytest.raises(ValueError, match="[Yy]AML|parse"):
            parse_agent_file(agent_file)

    def test_missing_required_name_raises(self, tmp_path: Path):
        """Test that missing name field raises validation error."""
        from pydantic import ValidationError

        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
description: "No name field"
version: "1.0.0"
---

# Content
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        with pytest.raises(ValidationError):
            parse_agent_file(agent_file)

    def test_empty_file_raises(self, tmp_path: Path):
        """Test that empty file raises ValueError."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_file = tmp_path / "agent.md"
        agent_file.write_text("")

        with pytest.raises(ValueError, match="empty|frontmatter"):
            parse_agent_file(agent_file)

    def test_only_frontmatter_delimiters_raises(self, tmp_path: Path):
        """Test that file with only --- markers raises error."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
---
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        # Should fail because name is required
        with pytest.raises((ValueError, Exception)):
            parse_agent_file(agent_file)


class TestParseAgentFileEdgeCases:
    """Edge case tests for parse_agent_file."""

    def test_whitespace_before_frontmatter_fails(self, tmp_path: Path):
        """Test that whitespace before frontmatter is handled correctly."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """
---
name: "Whitespace Before"
---

# Content
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        # Leading whitespace means no valid frontmatter
        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_triple_dash_in_content_not_confused(self, tmp_path: Path):
        """Test that --- in content doesn't confuse parser."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "Content With Dashes"
---

# Section

Here's some content.

---

This is a horizontal rule above.

---

And another one.
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, prompt_content = parse_agent_file(agent_file)

        assert metadata.name == "Content With Dashes"
        # The --- in content should be preserved
        assert prompt_content.count("---") == 2

    def test_empty_prompt_content_after_frontmatter(self, tmp_path: Path):
        """Test handling of file with no content after frontmatter."""
        from saber.inspect_ai.agents.registry.custom_agent import parse_agent_file

        agent_content = """---
name: "No Content Agent"
---
"""
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(agent_content)

        metadata, prompt_content = parse_agent_file(agent_file)

        assert metadata.name == "No Content Agent"
        assert prompt_content.strip() == ""
