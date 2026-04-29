"""Workspace scaffolding helpers for the SABER CLI."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from importlib.metadata import Distribution, PackageNotFoundError, distribution
from pathlib import Path
from textwrap import dedent
from urllib.parse import urlparse
from urllib.request import url2pathname

from saber.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_PROJECT_NAME = "saber-eval-workspace"
_EXAMPLE_DOMAIN_SLUG = "starter_demo"
_EXAMPLE_TASK_ID = "starter_demo_task_1"


@dataclass(frozen=True)
class SaberDependency:
    """A SABER dependency declaration for a generated uv workspace."""

    dependency: str
    uv_sources: tuple[str, ...] = ()


def create_eval_workspace(target_dir: Path, include_demo_domain: bool = True) -> Path:
    """Create a new SABER evaluation workspace.

    Args:
        target_dir: Destination directory for the workspace. The path must
            not already exist.
        include_demo_domain: When ``True``, populate the workspace with the
            starter demo domain and task scaffold.

    Returns:
        The absolute path to the created workspace directory.

    Raises:
        FileExistsError: If *target_dir* already exists.
        NotADirectoryError: If the destination parent is not a directory.
        RuntimeError: If the current SABER dependency cannot be resolved.
        OSError: If filesystem operations fail.
    """
    destination = _resolve_destination(target_dir)
    if destination.exists():
        kind = "directory" if destination.is_dir() else "file"
        raise FileExistsError(
            f"Target {kind} already exists at {destination}. Choose a new directory path for the workspace."
        )

    parent = destination.parent
    if parent.exists() and not parent.is_dir():
        raise NotADirectoryError(f"Workspace parent is not a directory: {parent}")
    parent.mkdir(parents=True, exist_ok=True)

    dependency = _resolve_current_saber_dependency()
    project_name = _normalize_project_name(destination.name)

    temp_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}-", dir=parent))
    try:
        _write_workspace(temp_root, project_name, dependency, include_demo_domain)
        temp_root.rename(destination)
    except OSError:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return destination


def _resolve_destination(target_dir: Path) -> Path:
    """Resolve a workspace destination against the current working directory."""
    expanded = target_dir.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (Path.cwd() / expanded).resolve()


def _normalize_project_name(raw_name: str) -> str:
    """Normalize a directory name into a valid project name."""
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw_name.strip().lower()).strip("-.")
    return normalized or _DEFAULT_PROJECT_NAME


def _resolve_current_saber_dependency() -> SaberDependency:
    """Resolve the SABER dependency pin for the generated workspace."""
    dist = _load_saber_distribution()
    direct_url_text = dist.read_text("direct_url.json") if dist is not None else None

    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid saber direct_url.json metadata")
        else:
            resolved = _dependency_from_direct_url(direct_url)
            if resolved is not None:
                return resolved

    version = dist.version if dist is not None else _load_local_project_version()
    if not version:
        raise RuntimeError("Unable to determine the current SABER package version.")

    return SaberDependency(dependency=f"saber=={version}")


def _load_saber_distribution() -> Distribution | None:
    """Load the installed SABER distribution metadata when available."""
    try:
        return distribution("saber")
    except PackageNotFoundError:
        return None


def _dependency_from_direct_url(direct_url: object) -> SaberDependency | None:
    """Build a workspace dependency from PEP 610 direct URL metadata."""
    if not isinstance(direct_url, Mapping):
        return None

    raw_url = direct_url.get("url")
    if not isinstance(raw_url, str) or not raw_url:
        return None

    subdirectory = direct_url.get("subdirectory")
    if isinstance(subdirectory, str) and subdirectory:
        subdirectory = subdirectory.strip()

    dir_info = direct_url.get("dir_info")
    if isinstance(dir_info, Mapping):
        local_path = _path_from_file_url(raw_url)
        if local_path is not None:
            source: dict[str, object] = {"path": str(local_path)}
            if dir_info.get("editable") is True:
                source["editable"] = True
            if subdirectory:
                source["subdirectory"] = subdirectory
            uv_sources = [_render_uv_source_line("saber", source)]
            uv_sources.extend(_load_extra_uv_sources(local_path))
            return SaberDependency(dependency="saber", uv_sources=tuple(uv_sources))

    vcs_info = direct_url.get("vcs_info")
    if isinstance(vcs_info, Mapping):
        vcs = vcs_info.get("vcs")
        if isinstance(vcs, str) and vcs == "git":
            git_source: dict[str, object] = {"git": raw_url}
            commit_id = vcs_info.get("commit_id")
            requested_revision = vcs_info.get("requested_revision")
            if isinstance(commit_id, str) and commit_id:
                git_source["rev"] = commit_id
            elif isinstance(requested_revision, str) and requested_revision:
                git_source["rev"] = requested_revision
            if subdirectory:
                git_source["subdirectory"] = subdirectory
            return SaberDependency(
                dependency="saber",
                uv_sources=(_render_uv_source_line("saber", git_source),),
            )

        return SaberDependency(dependency=f"saber @ {raw_url}")

    return SaberDependency(dependency=f"saber @ {raw_url}")


def _path_from_file_url(raw_url: str) -> Path | None:
    """Convert a file:// URL into a local filesystem path."""
    parsed = urlparse(raw_url)
    if parsed.scheme != "file":
        return None

    path_text = f"//{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
    return Path(url2pathname(path_text)).resolve()


def _load_extra_uv_sources(project_root: Path) -> tuple[str, ...]:
    """Load additional uv source pins required by the current SABER project."""
    pyproject_path = project_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return ()

    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError:
        logger.warning("Ignoring invalid pyproject.toml at %s", pyproject_path)
        return ()

    tool_data = pyproject.get("tool")
    if not isinstance(tool_data, Mapping):
        return ()
    uv_data = tool_data.get("uv")
    if not isinstance(uv_data, Mapping):
        return ()
    sources = uv_data.get("sources")
    if not isinstance(sources, Mapping):
        return ()

    rendered: list[str] = []
    inspect_ai_source = sources.get("inspect-ai")
    if isinstance(inspect_ai_source, Mapping):
        rendered.append(_render_uv_source_line("inspect-ai", inspect_ai_source))
    return tuple(rendered)


def _render_uv_source_line(package_name: str, source: Mapping[str, object]) -> str:
    """Render a single `[tool.uv.sources]` entry."""
    return f"{_toml_string(package_name)} = {_render_inline_toml_table(source)}"


def _render_inline_toml_table(source: Mapping[str, object]) -> str:
    """Render a TOML inline table containing simple scalar values."""
    parts: list[str] = []
    for key, value in source.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, str):
            rendered = _toml_string(value)
        elif isinstance(value, (int, float)):
            rendered = str(value)
        else:
            logger.debug("Skipping unsupported uv source value for key %s: %r", key, value)
            continue
        parts.append(f"{key} = {rendered}")
    return "{ " + ", ".join(parts) + " }"


def _toml_string(value: str) -> str:
    """Render a TOML-safe basic string."""
    return json.dumps(value)


def _load_local_project_version() -> str | None:
    """Load the local SABER project version from pyproject.toml when present."""
    project_root = _find_local_project_root()
    if project_root is None:
        return None

    pyproject_path = project_root / "pyproject.toml"
    try:
        pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return None

    project_data = pyproject.get("project")
    if not isinstance(project_data, Mapping):
        return None
    version = project_data.get("version")
    return version if isinstance(version, str) else None


def _find_local_project_root() -> Path | None:
    """Find the local SABER project root when running from source."""
    current = Path(__file__).resolve()
    for parent in current.parents:
        pyproject_path = parent / "pyproject.toml"
        if pyproject_path.is_file():
            return parent
    return None


def _write_workspace(root: Path, project_name: str, dependency: SaberDependency, include_demo_domain: bool) -> None:
    """Write all generated workspace files beneath *root*."""
    (root / "domains").mkdir(parents=True, exist_ok=True)

    files = {
        Path("pyproject.toml"): _render_pyproject(project_name, dependency),
        Path("README.md"): _render_readme(include_demo_domain),
    }
    if include_demo_domain:
        files.update(
            {
                Path("domains") / _EXAMPLE_DOMAIN_SLUG / f"{_EXAMPLE_DOMAIN_SLUG}.py": _render_domain_module(),
                Path("domains") / _EXAMPLE_DOMAIN_SLUG / "eval.yaml": _render_eval_yaml(),
                Path("domains") / _EXAMPLE_DOMAIN_SLUG / "tasks" / "global.yaml": _render_global_yaml(),
                Path("domains") / _EXAMPLE_DOMAIN_SLUG / "tasks" / "starter_task.yaml": _render_task_yaml(),
                Path("domains")
                / _EXAMPLE_DOMAIN_SLUG
                / "prompts"
                / "instructions"
                / "starter_demo.j2": _render_instruction_prompt(),
                Path("domains")
                / _EXAMPLE_DOMAIN_SLUG
                / "prompts"
                / "assistants"
                / "starter_assistant.j2": _render_assistant_prompt(),
            }
        )

    for relative_path, content in files.items():
        file_path = root / relative_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")


def _render_pyproject(project_name: str, dependency: SaberDependency) -> str:
    """Render the generated workspace's pyproject.toml."""
    lines = [
        "[project]",
        f"name = {_toml_string(project_name)}",
        'version = "0.1.0"',
        'description = "Vanilla SABER benchmark/evaluation workspace"',
        'readme = "README.md"',
        'requires-python = ">=3.11,<3.14"',
        "dependencies = [",
        f"    {_toml_string(dependency.dependency)},",
        "]",
    ]
    if dependency.uv_sources:
        lines.extend(["", "[tool.uv.sources]", *dependency.uv_sources])
    return "\n".join(lines) + "\n"


def _render_readme(include_demo_domain: bool) -> str:
    """Render the generated workspace README."""
    if not include_demo_domain:
        return dedent(
            """\
            # SABER Evaluation Workspace

            This workspace was created by `saber new-eval-workspace --no-demo-domain`.
            No starter demo domain was generated, so `domains/` is ready for your
            first benchmark domain.

            The generated `pyproject.toml` pins `saber` to the same source or version
            used to create this workspace, so `uv sync` will install a compatible base.

            ## Get started

            1. Install dependencies:

               ```bash
               uv sync
               ```

            2. Create your first domain under `domains/` using the layout and YAML
               snippets below.

            3. After adding a domain, confirm Inspect AI can discover it:

               ```bash
               uv run inspect list tasks | grep <your-domain-slug>
               ```

            ## Workspace layout

            ```text
            .
            ├── README.md
            ├── pyproject.toml
            └── domains/
            ```

            ## Create a new domain

            Create a directory under `domains/` that matches your domain slug, then add:

            ```text
            domains/my_domain/
            ├── my_domain.py
            ├── eval.yaml
            ├── prompts/
            │   ├── assistants/
            │   │   └── assistant.j2
            │   └── instructions/
            │       └── instruction.j2
            └── tasks/
                ├── global.yaml
                └── example_task.yaml
            ```

            Your `my_domain.py` entrypoint should call `saber.task.create_task()`:

            ```python
            from inspect_ai import Task, task

            from saber.task import create_task


            @task
            def my_domain(**kwargs: str | None) -> Task:
                return create_task(**kwargs)
            ```

            A minimal `eval.yaml` looks like this:

            ```yaml
            slug: my_domain
            name: "My Domain"
            description: "Short description of the benchmark domain."
            version: "0.1.0"
            ```

            In `tasks/global.yaml`, set the shared prompts and defaults:

            ```yaml
            global_defaults:
              prompts:
                instruction: "instructions/instruction.j2"
                assistant: "assistants/assistant.j2"
              max_steps: 10
            ```

            Optional domain extensions:

            - `setup.py` for data downloads or task generation hooks
            - `tools/` for domain-specific MCP tools
            - `scoring/` for custom scoring strategies

            ## Add a new task

            Add a YAML file under `domains/<your-domain>/tasks/` with one or more
            entries in a `tasks:` list. Each task should include:

            - `task_id`
            - `title`
            - `description`
            - any task-specific `initial_context`
            - optional `scoring` definitions

            A minimal task looks like this:

            ```yaml
            tasks:
              - task_id: example_task
                title: "Example task"
                description: "Answer the question using the provided context."
                initial_context:
                  question: "What value should be returned?"
                  answer: "example-value"
                scoring:
                  static:
                    submission:
                      target: submission
                      expected_answers:
                        - "example-value"
                      max_score: 1.0
            ```
            """
        )

    return dedent(
        f"""\
        # SABER Evaluation Workspace

        This workspace was created by `saber new-eval-workspace`. It starts with a
        minimal SABER domain you can copy, rename, and extend.

        The generated `pyproject.toml` pins `saber` to the same source or version
        used to create this workspace, so `uv sync` will install a compatible base.

        ## Get started

        1. Install dependencies:

           ```bash
           uv sync
           ```

        2. Confirm the sample domain is discoverable:

           ```bash
           uv run inspect list tasks | grep {_EXAMPLE_DOMAIN_SLUG}
           ```

        3. Run the sample domain:

           ```bash
           uv run inspect eval domains/{_EXAMPLE_DOMAIN_SLUG} --model openai/gpt-4.1-mini
           ```

        ## Workspace layout

        ```text
        .
        ├── README.md
        ├── pyproject.toml
        └── domains/
            └── {_EXAMPLE_DOMAIN_SLUG}/
                ├── {_EXAMPLE_DOMAIN_SLUG}.py
                ├── eval.yaml
                ├── prompts/
                │   ├── assistants/
                │   │   └── starter_assistant.j2
                │   └── instructions/
                │       └── starter_demo.j2
                └── tasks/
                    ├── global.yaml
                    └── starter_task.yaml
        ```

        ## Create a new domain

        1. Copy `domains/{_EXAMPLE_DOMAIN_SLUG}` to a new directory named after your
           domain slug.
        2. Rename `{_EXAMPLE_DOMAIN_SLUG}.py` so the file name matches the directory
           name, then update the `@task` function and `eval.yaml` slug.
        3. Edit `tasks/global.yaml` to set your shared prompts, defaults, tools, and
           optional permanent environment.
        4. Replace the starter prompts under `prompts/` with instructions tailored to
           your benchmark.

        Optional domain extensions:

        - `setup.py` for data downloads or task generation hooks
        - `tools/` for domain-specific MCP tools
        - `scoring/` for custom scoring strategies

        ## Add a new task

        Add another YAML file under `domains/<your-domain>/tasks/` with one or more
        entries in a `tasks:` list. Each task should include:

        - `task_id`
        - `title`
        - `description`
        - any task-specific `initial_context`
        - optional `scoring` definitions (the starter task uses static scoring)

        A minimal task looks like this:

        ```yaml
        tasks:
          - task_id: example_task
            title: "Example task"
            description: "Answer the question using the provided context."
            initial_context:
              question: "What value should be returned?"
              answer: "example-value"
            scoring:
              static:
                submission:
                  target: submission
                  expected_answers:
                    - "example-value"
                  max_score: 1.0
        ```

        ## Starter domain

        The included `{_EXAMPLE_DOMAIN_SLUG}` domain keeps everything deliberately
        small: one task, local prompts, no Docker images, and a static expected
        answer. Use it as the baseline pattern before adding sandboxes, tools, or
        custom scorers.
        """
    )


def _render_domain_module() -> str:
    """Render the sample Inspect AI domain module."""
    return dedent(
        f"""\
        \"\"\"Starter demo domain for Inspect AI.

        Run with:

            uv run inspect eval domains/{_EXAMPLE_DOMAIN_SLUG} --model openai/gpt-4.1-mini
        \"\"\"

        from inspect_ai import Task, task

        from saber.task import create_task


        @task
        def {_EXAMPLE_DOMAIN_SLUG}(**kwargs: str | None) -> Task:
            \"\"\"Create the starter demo Inspect AI task.\"\"\"
            return create_task(**kwargs)
        """
    )


def _render_eval_yaml() -> str:
    """Render the sample domain's eval.yaml."""
    return dedent(
        f"""\
        slug: {_EXAMPLE_DOMAIN_SLUG}
        name: "Starter Demo"
        description: >
          Minimal SABER starter domain that demonstrates the flat domain layout,
          prompt templates, and one static-scored task.
        version: "0.1.0"
        tags:
          - starter
          - example
          - minimal
        """
    )


def _render_global_yaml() -> str:
    """Render the sample domain's global task defaults."""
    return dedent(
        """\
        global_defaults:
          prompts:
            instruction: "instructions/starter_demo.j2"
            assistant: "assistants/starter_assistant.j2"
          max_steps: 5
        """
    )


def _render_task_yaml() -> str:
    """Render the sample domain's starter task."""
    return dedent(
        f"""\
        tasks:
          - task_id: {_EXAMPLE_TASK_ID}
            title: "Echo the demo hostname"
            description: "Read the initial context and return the demo hostname."
            initial_context:
              question: "What hostname should you return?"
              hostname: "aces-demo-host"
            scoring:
              static:
                submission:
                  target: submission
                  expected_answers:
                    - "aces-demo-host"
                  max_score: 1.0
        """
    )


def _render_instruction_prompt() -> str:
    """Render the starter instruction prompt."""
    return dedent(
        """\
        You are solving {{ title }}.

        Question:
        {{ initial_context.question }}

        Relevant context:
        - hostname: {{ initial_context.hostname }}

        Return only the hostname.
        """
    )


def _render_assistant_prompt() -> str:
    """Render the starter assistant prompt."""
    return "Be concise and answer with the final hostname only.\n"
