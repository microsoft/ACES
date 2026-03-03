"""Agent persona file parsing and models.

Parse Copilot agent persona files (YAML frontmatter + markdown body) into
strongly-typed Pydantic models.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class AgentToolsConfig(BaseModel):
    """Tool allowed/disallowed configuration from agent frontmatter."""

    model_config = ConfigDict(frozen=True)

    allowed: list[str] | None = Field(default=None)
    disallowed: list[str] | None = Field(default=None)


class AgentFileMetadata(BaseModel):
    """YAML frontmatter metadata from an agent.md file."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    description: str | None = Field(default=None)
    version: str | None = Field(default=None)
    model: str | None = Field(default=None)
    tools: AgentToolsConfig | None = Field(default=None)
    permission_mode: str | None = Field(default=None)
    max_turns: int | None = Field(default=None)
    skill_directories: list[str] | None = Field(default=None)


class CustomAgentConfig(BaseModel):
    """Copilot SDK custom agent config with Pydantic validation."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    display_name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    prompt: str = Field(..., min_length=1)
    tools: list[str] | None = Field(default=None)
    infer: bool = Field(default=True)

    def to_sdk_dict(self) -> dict[str, str | bool | list[str] | None]:
        """Convert to camelCase dict for Copilot SDK wire format.

        Always includes ``name``, ``prompt``, ``infer``.
        Only includes ``displayName``, ``description``, ``tools`` when not None.
        """
        result: dict[str, str | bool | list[str] | None] = {
            "name": self.name,
            "prompt": self.prompt,
            "infer": self.infer,
        }
        if self.display_name is not None:
            result["displayName"] = self.display_name
        if self.description is not None:
            result["description"] = self.description
        if self.tools is not None:
            result["tools"] = self.tools
        return result


# ── Frontmatter regex ────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n?(.*)",
    re.DOTALL,
)


def parse_agent_file(file_path: str | Path) -> tuple[AgentFileMetadata, str]:
    """Parse an agent ``.md`` file into metadata and prompt body.

    File format::

        ---
        name: my-agent
        description: A helpful agent
        tools:
          allowed: [tool1, tool2]
        ---
        # Agent Prompt Body
        You are a helpful agent...

    Args:
        file_path: Path to the agent markdown file.

    Returns:
        Tuple of (AgentFileMetadata, prompt_content_markdown).

    Raises:
        FileNotFoundError: If *file_path* does not exist.
        ValueError: If frontmatter is missing, empty, or malformed.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")

    text = path.read_text(encoding="utf-8")

    match = _FRONTMATTER_RE.match(text)
    if match is None:
        raise ValueError(
            f"Missing or malformed YAML frontmatter in {path}. Expected file to start with '---' delimiters."
        )

    raw_yaml = match.group(1)
    body = match.group(2).strip()

    data = yaml.safe_load(raw_yaml)
    if not isinstance(data, dict):
        raise ValueError(f"YAML frontmatter in {path} did not parse to a mapping.")

    try:
        metadata = AgentFileMetadata(**data)
    except ValidationError as exc:
        raise ValueError(f"Invalid agent metadata in {path}: {exc}") from exc
    return metadata, body
