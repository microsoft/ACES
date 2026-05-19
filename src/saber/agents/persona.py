# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Agent persona file parsing and models.

Parse Copilot agent persona files (YAML frontmatter + markdown body) into
strongly-typed Pydantic models.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AgentToolsConfig(BaseModel):
    """Tool allowed/disallowed configuration from agent frontmatter."""

    model_config = ConfigDict(frozen=True)

    allowed: list[str] | None = Field(default=None)
    disallowed: list[str] | None = Field(default=None)

    @field_validator("allowed", "disallowed", mode="before")
    @classmethod
    def _coerce_tool_list(cls, value: object) -> object:
        if value is None or isinstance(value, list):
            return value
        if isinstance(value, str):
            return [value]
        return value


class AgentFileMetadata(BaseModel):
    """YAML frontmatter metadata from an agent.md file."""

    model_config = ConfigDict(frozen=True, populate_by_name=True, extra="allow")

    name: str = Field(..., min_length=1)
    description: str | None = Field(default=None)
    version: str | None = Field(default=None)
    model: str | None = Field(default=None)
    tools: AgentToolsConfig | None = Field(default=None)
    permission_mode: str | None = Field(default=None)
    max_turns: int | None = Field(default=None)
    skill_directories: list[str] | None = Field(default=None)
    user_invocable: bool | None = Field(default=None, alias="user-invocable")

    @field_validator("tools", mode="before")
    @classmethod
    def _coerce_tools_config(cls, value: object) -> object:
        if value is None or isinstance(value, dict):
            return value
        if isinstance(value, str):
            return {"allowed": [value]}
        if isinstance(value, list):
            return {"allowed": value}
        return value


class CustomAgentConfig(BaseModel):
    """Copilot SDK custom agent config with Pydantic validation."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., min_length=1)
    display_name: str | None = Field(default=None)
    description: str | None = Field(default=None)
    prompt: str = Field(..., min_length=1)
    tools: list[str] | None = Field(default=None)
    mcp_servers: dict[str, object] | None = Field(default=None)
    infer: bool = Field(default=True)

    def to_sdk_dict(self) -> dict[str, object]:
        """Convert to the snake_case dict expected by Copilot SDK.

        Always includes ``name``, ``prompt``, ``infer``.
        Only includes optional fields when not None.
        """
        result: dict[str, object] = {
            "name": self.name,
            "prompt": self.prompt,
            "infer": self.infer,
        }
        if self.display_name is not None:
            result["display_name"] = self.display_name
        if self.description is not None:
            result["description"] = self.description
        if self.tools is not None:
            result["tools"] = self.tools
        if self.mcp_servers is not None:
            result["mcp_servers"] = self.mcp_servers
        return result


@dataclass(frozen=True)
class AgentDefinition:
    """Parsed agent definition from a markdown persona file."""

    path: Path
    relative_path: Path
    metadata: AgentFileMetadata
    prompt: str
    raw_content: str
    raw_metadata: dict[str, Any]

    @property
    def name(self) -> str:
        """Frontmatter agent name."""
        return self.metadata.name

    @property
    def aliases(self) -> tuple[str, ...]:
        """Names that can be used to select this agent."""
        stem = self.path.stem
        agent_stem = stem.removesuffix(".agent")
        return (
            self.metadata.name,
            self.path.name,
            stem,
            agent_stem,
            _normalize_agent_selector(self.metadata.name),
            _normalize_agent_selector(self.path.name),
            _normalize_agent_selector(stem),
            _normalize_agent_selector(agent_stem),
        )

    def to_custom_agent_config(self) -> CustomAgentConfig:
        """Convert this definition to a Copilot SDK custom agent config."""
        return CustomAgentConfig(
            name=self.metadata.name,
            display_name=self.metadata.name,
            description=self.metadata.description,
            prompt=self.prompt,
            tools=self.metadata.tools.allowed if self.metadata.tools else None,
            infer=self.metadata.user_invocable is not False,
        )


@dataclass(frozen=True)
class AgentBundle:
    """Resolved runtime agent bundle selected by the eval operator."""

    root: Path | None
    agents_dir: Path | None
    agents: tuple[AgentDefinition, ...]
    main_agent: AgentDefinition | None
    skills_dir: Path | None
    mcp_config: Path | None
    manifest_path: Path | None

    @property
    def has_agents(self) -> bool:
        """Whether this bundle contains one or more agent definitions."""
        return bool(self.agents)

    def custom_agent_configs(self) -> list[CustomAgentConfig]:
        """Return Copilot SDK custom agent configs for all parsed agents."""
        return [agent.to_custom_agent_config() for agent in self.agents]


# ── Frontmatter regex ────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(
    r"\A---[ \t]*\r?\n(.*?\r?\n)---[ \t]*\r?\n?(.*)",
    re.DOTALL,
)


def _parse_agent_text(text: str, path: Path) -> tuple[dict[str, Any], AgentFileMetadata, str]:
    """Parse raw agent markdown text."""
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
    return data, metadata, body


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

    _raw_metadata, metadata, body = _parse_agent_text(path.read_text(encoding="utf-8"), path)
    return metadata, body


def parse_agent_definition(file_path: str | Path, *, root: Path | None = None) -> AgentDefinition:
    """Parse an agent file into an :class:`AgentDefinition`.

    Args:
        file_path: Agent markdown file path.
        root: Optional root used to compute the sandbox relative path.

    Returns:
        Parsed agent definition including raw file content.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Agent file not found: {path}")
    raw_content = path.read_text(encoding="utf-8")
    raw_metadata, metadata, body = _parse_agent_text(raw_content, path)
    relative_path = _safe_relative_path(path, root.resolve() if root else path.parent)
    return AgentDefinition(
        path=path,
        relative_path=relative_path,
        metadata=metadata,
        prompt=body,
        raw_content=raw_content,
        raw_metadata=raw_metadata,
    )


def load_agent_bundle(
    *,
    agent_bundle: str | Path | None = None,
    persona_file: str | Path | None = None,
    agents_dir: str | Path | None = None,
    main_agent: str | None = None,
    skills_dir: str | Path | None = None,
    mcp_config: str | Path | None = None,
) -> AgentBundle:
    """Resolve operator-provided runtime agent bundle options.

    This is intentionally a SABER runtime concern. Exported scenario bundles
    should not carry these paths or copy these assets.
    """
    bundle_root = _optional_dir(agent_bundle, "Agent bundle")
    manifest_path = _find_manifest(bundle_root)
    manifest = _read_manifest(manifest_path)

    resolved_agents_dir = _optional_dir(agents_dir, "Agents directory")
    if resolved_agents_dir is None and bundle_root is not None:
        resolved_agents_dir = bundle_root

    resolved_persona = _optional_file(persona_file, "Persona file")
    if resolved_persona is None and bundle_root is not None:
        resolved_persona = _manifest_agent_file(bundle_root, manifest)

    agent_files = _discover_agent_files(resolved_agents_dir, resolved_persona)
    parse_root = resolved_agents_dir or (resolved_persona.parent if resolved_persona else bundle_root)
    agents = tuple(parse_agent_definition(path, root=parse_root) for path in agent_files)

    resolved_main = _select_main_agent(
        agents=agents,
        explicit=main_agent,
        manifest_agent=manifest.get("agent") if isinstance(manifest.get("agent"), str) else None,
        persona_file=resolved_persona,
    )

    resolved_skills = _optional_dir(skills_dir, "Skills directory")
    if resolved_skills is None and bundle_root is not None and (bundle_root / "skills").is_dir():
        resolved_skills = (bundle_root / "skills").resolve()

    resolved_mcp = _optional_file(mcp_config, "MCP config")
    if resolved_mcp is None and bundle_root is not None and (bundle_root / ".mcp.json").is_file():
        resolved_mcp = (bundle_root / ".mcp.json").resolve()

    return AgentBundle(
        root=bundle_root,
        agents_dir=resolved_agents_dir,
        agents=agents,
        main_agent=resolved_main,
        skills_dir=resolved_skills,
        mcp_config=resolved_mcp,
        manifest_path=manifest_path,
    )


def _optional_dir(value: str | Path | None, label: str) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    return path


def _optional_file(value: str | Path | None, label: str) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _find_manifest(bundle_root: Path | None) -> Path | None:
    if bundle_root is None:
        return None
    path = bundle_root / "manifest.json"
    return path.resolve() if path.is_file() else None


def _read_manifest(manifest_path: Path | None) -> dict[str, Any]:
    if manifest_path is None:
        return {}
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _manifest_agent_file(bundle_root: Path, manifest: dict[str, Any]) -> Path | None:
    agent_value = manifest.get("agent")
    if not isinstance(agent_value, str) or not agent_value.strip():
        return None
    path = (bundle_root / agent_value).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Manifest agent file does not exist: {path}")
    _safe_relative_path(path, bundle_root)
    return path


def _discover_agent_files(agents_dir: Path | None, persona_file: Path | None) -> list[Path]:
    discovered: list[Path] = []
    if agents_dir is not None:
        discovered.extend(sorted(path.resolve() for path in agents_dir.glob("*.agent.md") if path.is_file()))
        if not discovered:
            discovered.extend(
                sorted(
                    path.resolve()
                    for path in agents_dir.glob("*.md")
                    if path.is_file() and path.name.lower() not in {"readme.md", "license.md"}
                )
            )

    if persona_file is not None:
        discovered.append(persona_file.resolve())

    unique: dict[Path, Path] = {}
    for path in discovered:
        unique[path] = path
    return list(unique.values())


def _select_main_agent(
    *,
    agents: tuple[AgentDefinition, ...],
    explicit: str | None,
    manifest_agent: str | None,
    persona_file: Path | None,
) -> AgentDefinition | None:
    if not agents:
        return None
    if explicit:
        return _find_agent(agents, explicit)
    if manifest_agent:
        try:
            return _find_agent(agents, manifest_agent)
        except ValueError:
            pass
    if persona_file is not None:
        try:
            return _find_agent(agents, persona_file.name)
        except ValueError:
            pass
    user_invocable = [agent for agent in agents if agent.metadata.user_invocable is not False]
    if len(user_invocable) == 1:
        return user_invocable[0]
    return agents[0]


def _find_agent(agents: tuple[AgentDefinition, ...], selector: str) -> AgentDefinition:
    normalized = _normalize_agent_selector(selector)
    for agent in agents:
        if selector in agent.aliases or normalized in agent.aliases:
            return agent
    available = ", ".join(agent.metadata.name for agent in agents)
    raise ValueError(f"Agent '{selector}' not found in bundle. Available agents: {available}")


def _normalize_agent_selector(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _safe_relative_path(path: Path, root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Path {resolved_path} escapes root {resolved_root}") from exc
    if any(part == ".." for part in relative.parts):
        raise ValueError(f"Unsafe relative path: {relative}")
    return relative
