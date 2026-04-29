"""Tests for agent persona file parsing and models."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from saber.agents.persona import (
    AgentFileMetadata,
    AgentToolsConfig,
    CustomAgentConfig,
    load_agent_bundle,
    parse_agent_file,
)

# ── AgentFileMetadata ────────────────────────────────────────────────


class TestAgentFileMetadata:
    """Tests for AgentFileMetadata model."""

    def test_agent_file_metadata_required_name(self) -> None:
        """Missing name raises ValidationError."""
        with pytest.raises(ValidationError):
            AgentFileMetadata()  # type: ignore[call-arg]

    def test_agent_file_metadata_frozen(self) -> None:
        """Mutation of frozen model raises ValidationError."""
        meta = AgentFileMetadata(name="test-agent")
        with pytest.raises(ValidationError):
            meta.name = "mutated"  # type: ignore[misc]

    def test_agent_file_metadata_optional_fields(self) -> None:
        """All optional fields default to None."""
        meta = AgentFileMetadata(name="test-agent")
        assert meta.name == "test-agent"
        assert meta.description is None
        assert meta.version is None
        assert meta.model is None
        assert meta.tools is None
        assert meta.permission_mode is None
        assert meta.max_turns is None
        assert meta.skill_directories is None

    def test_agent_file_metadata_with_skill_directories(self) -> None:
        """skill_directories list of strings parsed correctly."""
        meta = AgentFileMetadata(
            name="skilled-agent",
            skill_directories=["/skills/a", "/skills/b"],
        )
        assert meta.skill_directories == ["/skills/a", "/skills/b"]

    def test_agent_file_metadata_accepts_list_tools_and_user_invocable_alias(self) -> None:
        """MTP-style list tools and user-invocable metadata are accepted."""
        meta = AgentFileMetadata(
            name="recon-agent",
            tools=["agent"],
            **{"user-invocable": False, "owner": "security"},
        )

        assert meta.tools is not None
        assert meta.tools.allowed == ["agent"]
        assert meta.user_invocable is False
        assert meta.model_extra == {"owner": "security"}


# ── AgentToolsConfig ────────────────────────────────────────────────


class TestAgentToolsConfig:
    """Tests for AgentToolsConfig model."""

    def test_agent_tools_config_frozen(self) -> None:
        """Mutation of frozen model raises ValidationError."""
        cfg = AgentToolsConfig(allowed=["tool1"])
        with pytest.raises(ValidationError):
            cfg.allowed = ["tool2"]  # type: ignore[misc]

    def test_agent_tools_config_accepts_string_values(self) -> None:
        """Single string tool values are normalized to one-item lists."""
        cfg = AgentToolsConfig(allowed="agent", disallowed="web")  # type: ignore[arg-type]

        assert cfg.allowed == ["agent"]
        assert cfg.disallowed == ["web"]


# ── CustomAgentConfig ───────────────────────────────────────────────


class TestCustomAgentConfig:
    """Tests for CustomAgentConfig model."""

    def test_custom_agent_config_to_sdk_dict(self) -> None:
        """snake_case SDK keys produced; optional fields included when set."""
        cfg = CustomAgentConfig(
            name="my-agent",
            display_name="My Agent",
            description="A helpful agent",
            prompt="You are helpful.",
        )
        sdk = cfg.to_sdk_dict()
        assert sdk["name"] == "my-agent"
        assert sdk["display_name"] == "My Agent"
        assert sdk["description"] == "A helpful agent"
        assert sdk["prompt"] == "You are helpful."
        assert sdk["infer"] is True
        # tools not set → key absent
        assert "tools" not in sdk

    def test_custom_agent_config_to_sdk_dict_with_tools(self) -> None:
        """tools list preserved in SDK dict."""
        cfg = CustomAgentConfig(
            name="tool-agent",
            prompt="Use tools.",
            tools=["tool_a", "tool_b"],
        )
        sdk = cfg.to_sdk_dict()
        assert sdk["tools"] == ["tool_a", "tool_b"]

    def test_custom_agent_config_to_sdk_dict_no_optional(self) -> None:
        """Only name, prompt, infer present when no optional fields set."""
        cfg = CustomAgentConfig(name="bare", prompt="Bare prompt.")
        sdk = cfg.to_sdk_dict()
        assert set(sdk.keys()) == {"name", "prompt", "infer"}

    def test_custom_agent_config_to_sdk_dict_with_mcp_servers(self) -> None:
        """Agent-scoped MCP servers use SDK snake_case key names."""
        cfg = CustomAgentConfig(
            name="mcp-agent",
            prompt="Use MCP.",
            mcp_servers={"Azure": {"type": "stdio", "command": "npx"}},
        )

        sdk = cfg.to_sdk_dict()

        assert sdk["mcp_servers"] == {"Azure": {"type": "stdio", "command": "npx"}}


# ── parse_agent_file ────────────────────────────────────────────────


class TestParseAgentFile:
    """Tests for parse_agent_file function."""

    def test_parse_agent_file_valid(self, tmp_path: Path) -> None:
        """Full round-trip: frontmatter + body parsed correctly."""
        content = textwrap.dedent("""\
            ---
            name: test-agent
            description: A test agent
            version: "1.0"
            model: gpt-4
            tools:
              allowed:
                - tool1
                - tool2
            permission_mode: auto
            max_turns: 5
            ---
            # Agent Prompt

            You are a helpful agent.
        """)
        agent_file = tmp_path / "agent.md"
        agent_file.write_text(content)

        meta, body = parse_agent_file(agent_file)

        assert meta.name == "test-agent"
        assert meta.description == "A test agent"
        assert meta.version == "1.0"
        assert meta.model == "gpt-4"
        assert meta.tools is not None
        assert meta.tools.allowed == ["tool1", "tool2"]
        assert meta.tools.disallowed is None
        assert meta.permission_mode == "auto"
        assert meta.max_turns == 5
        assert "# Agent Prompt" in body
        assert "You are a helpful agent." in body

    def test_parse_agent_file_no_frontmatter(self, tmp_path: Path) -> None:
        """File without frontmatter delimiters raises ValueError."""
        agent_file = tmp_path / "no_front.md"
        agent_file.write_text("# Just markdown\nNo frontmatter here.")

        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_parse_agent_file_empty(self, tmp_path: Path) -> None:
        """Empty file raises ValueError."""
        agent_file = tmp_path / "empty.md"
        agent_file.write_text("")

        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_parse_agent_file_not_found(self) -> None:
        """Non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_agent_file("/nonexistent/path/agent.md")

    def test_parse_agent_file_with_skill_directories(self, tmp_path: Path) -> None:
        """skill_directories populated from frontmatter."""
        content = textwrap.dedent("""\
            ---
            name: skilled-agent
            skill_directories:
              - /skills/a
              - /skills/b
            ---
            # Prompt body
        """)
        agent_file = tmp_path / "skilled.md"
        agent_file.write_text(content)

        meta, body = parse_agent_file(agent_file)
        assert meta.skill_directories == ["/skills/a", "/skills/b"]
        assert "# Prompt body" in body

    def test_parse_agent_file_with_list_tools_and_user_invocable(self, tmp_path: Path) -> None:
        """Native MTP-style agent frontmatter parses without compatibility copies."""
        content = textwrap.dedent("""\
            ---
            name: fingerprint-agent
            tools:
              - agent
            user-invocable: false
            ---
            # Fingerprint Agent
        """)
        agent_file = tmp_path / "fingerprint.agent.md"
        agent_file.write_text(content)

        meta, body = parse_agent_file(agent_file)

        assert meta.tools is not None
        assert meta.tools.allowed == ["agent"]
        assert meta.user_invocable is False
        assert "# Fingerprint Agent" in body

    def test_parse_agent_file_unclosed_frontmatter(self, tmp_path: Path) -> None:
        """Unclosed frontmatter (missing closing ---) raises ValueError."""
        content = textwrap.dedent("""\
            ---
            name: broken-agent
            description: unclosed
            # Prompt body
        """)
        agent_file = tmp_path / "unclosed.md"
        agent_file.write_text(content)

        with pytest.raises(ValueError, match="frontmatter"):
            parse_agent_file(agent_file)

    def test_parse_agent_file_missing_name(self, tmp_path: Path) -> None:
        """Valid YAML frontmatter but no name field raises ValueError."""
        content = textwrap.dedent("""\
            ---
            description: no name here
            version: "1.0"
            ---
            # Prompt body
        """)
        agent_file = tmp_path / "no_name.md"
        agent_file.write_text(content)

        with pytest.raises(ValueError, match="Invalid agent metadata"):
            parse_agent_file(agent_file)

    def test_parse_agent_file_frontmatter_not_a_dict(self, tmp_path: Path) -> None:
        """Frontmatter that is a plain string raises ValueError."""
        content = textwrap.dedent("""\
            ---
            just a string
            ---
            # Prompt body
        """)
        agent_file = tmp_path / "string_front.md"
        agent_file.write_text(content)

        with pytest.raises(ValueError, match="did not parse to a mapping"):
            parse_agent_file(agent_file)


class TestLoadAgentBundle:
    """Runtime agent bundle resolution."""

    def test_load_agent_bundle_from_manifest(self, tmp_path: Path) -> None:
        """Bundle root discovers agents, skills, MCP config, and manifest main agent."""
        bundle_dir = tmp_path / "recon-agent"
        bundle_dir.mkdir()
        (bundle_dir / "skills").mkdir()
        (bundle_dir / "skills" / "README.md").write_text("# Skill")
        (bundle_dir / ".mcp.json").write_text(
            json.dumps({"mcpServers": {"Azure": {"type": "stdio", "command": "npx", "tools": ["*"]}}})
        )
        (bundle_dir / "manifest.json").write_text(json.dumps({"agent": "recon.agent.md"}))
        (bundle_dir / "recon.agent.md").write_text(
            textwrap.dedent("""\
                ---
                name: Recon Agent
                tools:
                  - agent
                ---
                Main recon prompt.
            """)
        )
        (bundle_dir / "fingerprint.agent.md").write_text(
            textwrap.dedent("""\
                ---
                name: Fingerprint Agent
                user-invocable: false
                ---
                Fingerprint prompt.
            """)
        )

        bundle = load_agent_bundle(agent_bundle=bundle_dir)

        assert bundle.root == bundle_dir.resolve()
        assert bundle.main_agent is not None
        assert bundle.main_agent.name == "Recon Agent"
        assert bundle.skills_dir == (bundle_dir / "skills").resolve()
        assert bundle.mcp_config == (bundle_dir / ".mcp.json").resolve()
        assert {agent.name for agent in bundle.agents} == {"Recon Agent", "Fingerprint Agent"}

        configs = {cfg.name: cfg.to_sdk_dict() for cfg in bundle.custom_agent_configs()}
        assert configs["Recon Agent"]["tools"] == ["agent"]
        assert configs["Fingerprint Agent"]["infer"] is False

    def test_load_agent_bundle_explicit_main_agent_selector(self, tmp_path: Path) -> None:
        """Explicit main_agent can select by normalized name or filename."""
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "attack-path.agent.md").write_text(
            textwrap.dedent("""\
                ---
                name: Attack Path Agent
                ---
                Attack path prompt.
            """)
        )
        (agents_dir / "report.agent.md").write_text(
            textwrap.dedent("""\
                ---
                name: Report Agent
                ---
                Report prompt.
            """)
        )

        bundle = load_agent_bundle(agents_dir=agents_dir, main_agent="attack-path")

        assert bundle.main_agent is not None
        assert bundle.main_agent.name == "Attack Path Agent"

    def test_load_agent_bundle_missing_main_agent_raises(self, tmp_path: Path) -> None:
        """Invalid explicit main agent selectors fail fast."""
        (tmp_path / "only.agent.md").write_text(
            textwrap.dedent("""\
                ---
                name: Only Agent
                ---
                Prompt.
            """)
        )

        with pytest.raises(ValueError, match="not found in bundle"):
            load_agent_bundle(agents_dir=tmp_path, main_agent="missing")
