"""Custom agent configuration models.

This module provides Pydantic models for custom Copilot agent personas,
matching the Copilot SDK's CustomAgentConfig TypedDict with strong
validation and SABER-specific tool mapping.

Classes:
    AgentToolsConfig: Tool allowed/disallowed configuration from agent frontmatter.
    AgentFileMetadata: YAML frontmatter parsed from agent.md files.
    CustomAgentConfig: Full configuration for a custom Copilot agent persona.

Functions:
    parse_agent_file: Parse an agent.md file and extract metadata and prompt.
    map_tools_to_saber: Map Claude tool names to SABER MCP tool names.

Constants:
    CLAUDE_TO_SABER_TOOL_MAP: Mapping from Claude tool names to SABER MCP tools.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

# Tool name mapping from Claude Code tools to SABER MCP tools
CLAUDE_TO_SABER_TOOL_MAP: dict[str, str | None] = {
    "Bash": "bash",
    "Read": "view",
    "Write": "write",
    "Edit": "edit",
    "WebFetch": "fetch_webpage",
    "WebSearch": "web_search",
    "Grep": "grep",
    "Glob": "glob",
    "LS": "view",
    # Skills are handled via skill_directories parameter, not as individual tools
    "Skill": None,
}


class AgentToolsConfig(BaseModel):
    """Tool configuration from agent frontmatter.

    Specifies which tools are allowed or disallowed for the agent.
    When both are None, all tools are available to the agent.

    Attributes:
        allowed: List of tool names that the agent can use.
            If specified, only these tools are available.
        disallowed: List of tool names that the agent cannot use.
            All other tools remain available.

    Example:
        >>> config = AgentToolsConfig(allowed=["Bash", "Read", "Write"])
        >>> config.allowed
        ['Bash', 'Read', 'Write']
    """

    model_config = ConfigDict(frozen=True)

    allowed: list[str] | None = Field(
        default=None,
        description="List of tool names the agent is allowed to use",
    )
    disallowed: list[str] | None = Field(
        default=None,
        description="List of tool names the agent is not allowed to use",
    )


class AgentFileMetadata(BaseModel):
    """YAML frontmatter metadata from an agent.md file.

    This model represents the structured metadata extracted from the
    YAML frontmatter section of an agent markdown file.

    Attributes:
        name: The agent's unique identifier name.
        description: Brief description of what the agent does.
        version: Version string for the agent definition.
        model: Preferred model for this agent (e.g., "claude-sonnet-4-5-20250929").
        tools: Tool restrictions configuration.
        permission_mode: Permission level for agent actions.
        max_turns: Maximum conversation turns allowed.

    Example:
        >>> metadata = AgentFileMetadata(
        ...     name="Red Team Agent",
        ...     description="Execute security testing tasks",
        ...     tools=AgentToolsConfig(allowed=["Bash", "Read"]),
        ... )
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Unique name/identifier for the agent",
    )
    description: str | None = Field(
        default=None,
        description="Brief description of what the agent does",
    )
    version: str | None = Field(
        default=None,
        description="Version string for the agent definition",
    )
    model: str | None = Field(
        default=None,
        description="Preferred LLM model for this agent",
    )
    tools: AgentToolsConfig | None = Field(
        default=None,
        description="Tool restrictions (allowed/disallowed lists)",
    )
    permission_mode: str | None = Field(
        default=None,
        description="Permission mode for agent actions",
    )
    max_turns: int | None = Field(
        default=None,
        description="Maximum number of conversation turns",
    )
    skill_directories: list[str] | None = Field(
        default=None,
        description="Directories containing skill files to load",
    )


class CustomAgentConfig(BaseModel):
    """Configuration for a custom Copilot agent persona.

    Mirrors the Copilot SDK's CustomAgentConfig TypedDict but with
    Pydantic validation and SABER-specific tool mapping support.

    Attributes:
        name: Unique identifier for the custom agent.
        display_name: Human-readable name shown in the UI.
        description: Brief description of what the agent does.
        prompt: The agent's system prompt/persona instructions.
        tools: List of tool names the agent can use (None means all tools).
        infer: Whether the agent is available for model inference.

    Example:
        >>> config = CustomAgentConfig(
        ...     name="red-team-agent",
        ...     display_name="Red Team Agent",
        ...     prompt="You are a security researcher.",
        ...     tools=["bash", "read_file"],
        ... )
        >>> config.to_sdk_dict()
        {'name': 'red-team-agent', 'displayName': 'Red Team Agent', ...}
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        description="Unique identifier for the custom agent",
    )
    display_name: str | None = Field(
        default=None,
        description="Human-readable display name for the UI",
    )
    description: str | None = Field(
        default=None,
        description="Brief description of what the agent does",
    )
    prompt: str = Field(
        ...,
        min_length=1,
        description="The agent's system prompt/persona instructions",
    )
    tools: list[str] | None = Field(
        default=None,
        description="Tool restrictions - None means all tools are available",
    )
    infer: bool = Field(
        default=True,
        description="Whether the agent is available for model inference",
    )

    def to_sdk_dict(self, tool_map: dict[str, str] | None = None) -> dict[str, object]:
        """Convert to Copilot SDK wire format with optional tool mapping.

        Converts the Pydantic model to a dictionary matching the Copilot SDK's
        expected format with camelCase keys. Optional fields are only included
        if they have non-None values.

        Args:
            tool_map: Optional mapping from source tool names (e.g., Claude tool names
                like "Bash", "Read") to target SABER MCP tool names (e.g., "bash",
                "read_file"). If provided and tools is not None, tool names will
                be mapped using this dictionary.

        Returns:
            Dictionary with camelCase keys matching Copilot SDK TypedDict format:
            - name: Agent identifier
            - displayName: (optional) Display name for UI
            - description: (optional) Agent description
            - prompt: System prompt
            - tools: (optional) List of mapped tool names
            - infer: Boolean for inference availability

        Example:
            >>> config = CustomAgentConfig(
            ...     name="test-agent",
            ...     prompt="Test",
            ...     tools=["Bash", "Read"],
            ... )
            >>> config.to_sdk_dict(tool_map={"Bash": "bash", "Read": "read_file"})
            {'name': 'test-agent', 'prompt': 'Test', 'tools': ['bash', 'read_file'], 'infer': True}
        """
        result: dict[str, object] = {
            "name": self.name,
            "prompt": self.prompt,
            "infer": self.infer,
        }

        if self.display_name is not None:
            result["displayName"] = self.display_name

        if self.description is not None:
            result["description"] = self.description

        if self.tools is not None:
            if tool_map is not None:
                # Map tool names using the provided mapping, keeping original if not found
                mapped_tools = [tool_map.get(tool, tool) for tool in self.tools]
                result["tools"] = mapped_tools
            else:
                result["tools"] = list(self.tools)

        return result


def parse_agent_file(file_path: str | Path) -> tuple[AgentFileMetadata, str]:
    """Parse an agent.md file and extract YAML frontmatter and prompt content.

    Reads an agent markdown file, extracts the YAML frontmatter between
    ``---`` markers, and returns the parsed metadata along with the
    remaining markdown content as the prompt.

    Args:
        file_path: Path to the agent.md file. Can be a string or Path object.

    Returns:
        A tuple of (metadata, prompt_content) where:
        - metadata: AgentFileMetadata parsed from YAML frontmatter
        - prompt_content: The markdown body after the frontmatter

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file has no valid YAML frontmatter, or if the
            frontmatter is malformed or cannot be parsed.

    Example:
        >>> metadata, prompt = parse_agent_file("agents/red-team/agent.md")
        >>> metadata.name
        'Red Team Agent'
        >>> prompt[:50]
        '# Red Team Agent\\n\\nExecute tasks directly...'
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {file_path}")

    content = path.read_text(encoding="utf-8")

    if not content.strip():
        raise ValueError(f"Agent file is empty: {file_path}")

    # Check for valid frontmatter: must start with ---
    if not content.startswith("---"):
        raise ValueError(f"Agent file must start with YAML frontmatter (---): {file_path}")

    # Find the closing --- of frontmatter
    # Skip the first --- and find the next one
    end_marker_pos = content.find("---", 3)
    if end_marker_pos == -1:
        raise ValueError(f"Agent file has unclosed YAML frontmatter (missing closing ---): {file_path}")

    # Extract frontmatter YAML (between the two ---)
    frontmatter_yaml = content[3:end_marker_pos].strip()

    # Extract the content after the frontmatter
    prompt_content = content[end_marker_pos + 3 :].strip()

    # Parse YAML frontmatter
    try:
        frontmatter_data = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML frontmatter in {file_path}: {e}") from e

    # Handle empty frontmatter
    if frontmatter_data is None:
        frontmatter_data = {}

    # Parse tools if present
    tools_data = frontmatter_data.get("tools")
    tools_config = None
    if tools_data is not None:
        tools_config = AgentToolsConfig(
            allowed=tools_data.get("allowed"),
            disallowed=tools_data.get("disallowed"),
        )

    # Create metadata (will validate required fields like name)
    metadata = AgentFileMetadata(
        name=frontmatter_data.get("name"),
        description=frontmatter_data.get("description"),
        version=frontmatter_data.get("version"),
        model=frontmatter_data.get("model"),
        tools=tools_config,
        permission_mode=frontmatter_data.get("permission_mode"),
        max_turns=frontmatter_data.get("max_turns"),
    )

    return metadata, prompt_content


def map_tools_to_saber(claude_tools: list[str]) -> list[str]:
    """Map Claude tool names to SABER MCP tool names.

    Converts a list of Claude Code tool names (e.g., "Bash", "Read", "Write")
    to their corresponding SABER MCP tool names (e.g., "bash", "read_file",
    "write_file"). Tools not in the mapping are passed through unchanged,
    allowing MCP-style tools and custom tools to work.

    The "Skill" tool is filtered out because skills are handled via the
    ``skill_directories`` parameter rather than as individual tools.

    Args:
        claude_tools: List of tool names from agent configuration.
            Can include Claude tool names, MCP-style tool names
            (e.g., "mcp__server__tool"), or custom tool names.

    Returns:
        List of SABER MCP tool names. The order is preserved.
        Tools mapping to None (like "Skill") are filtered out.

    Example:
        >>> map_tools_to_saber(["Bash", "Read", "Write"])
        ['bash', 'read_file', 'write_file']

        >>> map_tools_to_saber(["Bash", "Skill", "mcp__server__tool"])
        ['bash', 'mcp__server__tool']
    """
    result: list[str] = []

    for tool in claude_tools:
        if tool in CLAUDE_TO_SABER_TOOL_MAP:
            mapped = CLAUDE_TO_SABER_TOOL_MAP[tool]
            # Filter out None (e.g., Skill tool)
            if mapped is not None:
                result.append(mapped)
        else:
            # Unknown tools pass through unchanged
            result.append(tool)

    return result
