"""Tests for ConfigLoader — YAML config loading with 3-level inheritance."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from saber.config.loader import ConfigLoader, load_domain_config
from saber.config.models import DomainConfig, TaskConfig

# ── Helpers ─────────────────────────────────────────────────────────


def _write_yaml(path: Path, data: object) -> None:
    """Write a Python object as YAML to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False))


def _minimal_task(task_id: str = "task_1", **overrides: object) -> dict[str, object]:
    """Return a minimal valid task dict that passes TaskConfig validation."""
    base: dict[str, object] = {
        "task_id": task_id,
        "title": task_id,
        "description": f"Description for {task_id}",
    }
    base.update(overrides)
    return base


def _minimal_global_defaults() -> dict[str, object]:
    """Global defaults that supply prompts (required by TaskConfig)."""
    return {
        "prompts": {
            "instruction": "instructions/demo.j2",
            "assistant": "assistants/demo.j2",
            "submit": "submits/demo.j2",
            "continue": "continues/demo.j2",
        },
    }


def _make_domain(
    tmp_path: Path,
    *,
    global_yaml: dict[str, object] | None = None,
    tasks: dict[str, list[dict[str, object]]] | None = None,
    shared_yamls: dict[str, dict[str, object]] | None = None,
) -> Path:
    """Build a full domain directory structure under *tmp_path*.

    Args:
        global_yaml: Contents for tasks/global.yaml.
        tasks: Mapping of relative-path (e.g. "incident_5/inc5_1.yaml")
               to list of task dicts placed under the ``tasks:`` key.
        shared_yamls: Mapping of subdirectory name (e.g. "incident_5")
                      to the shared.yaml contents dict for that subdir.

    Returns:
        The domain root path (``tmp_path``).
    """
    tasks_dir = tmp_path / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompts").mkdir(exist_ok=True)

    if global_yaml is not None:
        _write_yaml(tasks_dir / "global.yaml", global_yaml)

    if shared_yamls:
        for subdir, content in shared_yamls.items():
            _write_yaml(tasks_dir / subdir / "shared.yaml", content)

    if tasks:
        for rel_path, task_list in tasks.items():
            _write_yaml(tasks_dir / rel_path, {"tasks": task_list})

    return tmp_path


# ── _load_global_defaults tests ─────────────────────────────────────


class TestLoadGlobalDefaults:
    """Tests for ConfigLoader._load_global_defaults."""

    def test_returns_global_defaults_key(self, tmp_path: Path) -> None:
        """When global.yaml has a 'global_defaults' key, return its contents."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {"prompts": {"instruction": "test.j2"}},
            },
        )
        loader = ConfigLoader(domain)
        result = loader._load_global_defaults()
        assert result == {"prompts": {"instruction": "test.j2"}}

    def test_returns_empty_when_no_global_yaml(self, tmp_path: Path) -> None:
        """When global.yaml is missing, return empty dict."""
        domain = _make_domain(tmp_path)
        loader = ConfigLoader(domain)
        assert loader._load_global_defaults() == {}

    def test_returns_empty_when_global_yaml_has_no_defaults_key(self, tmp_path: Path) -> None:
        """When global.yaml exists but lacks 'global_defaults', return {}."""
        domain = _make_domain(tmp_path, global_yaml={"domain": "test", "permanent_environment": "env1"})
        loader = ConfigLoader(domain)
        assert loader._load_global_defaults() == {}

    def test_returns_empty_when_global_yaml_is_empty(self, tmp_path: Path) -> None:
        """An empty global.yaml file returns {}."""
        domain = _make_domain(tmp_path)
        global_path = domain / "tasks" / "global.yaml"
        global_path.write_text("")
        loader = ConfigLoader(domain)
        assert loader._load_global_defaults() == {}


# ── _load_shared_context tests ──────────────────────────────────────


class TestLoadSharedContext:
    """Tests for ConfigLoader._load_shared_context."""

    def test_returns_dict_when_shared_exists(self, tmp_path: Path) -> None:
        """When shared.yaml exists in task_dir, return its contents."""
        domain = _make_domain(
            tmp_path,
            shared_yamls={"subdir": {"initial_context": {"key": "value"}}},
        )
        loader = ConfigLoader(domain)
        task_dir = domain / "tasks" / "subdir"
        result = loader._load_shared_context(task_dir)
        assert result == {"initial_context": {"key": "value"}}

    def test_returns_none_when_shared_missing(self, tmp_path: Path) -> None:
        """When shared.yaml doesn't exist, return None."""
        domain = _make_domain(tmp_path)
        loader = ConfigLoader(domain)
        task_dir = domain / "tasks" / "subdir"
        task_dir.mkdir(parents=True, exist_ok=True)
        assert loader._load_shared_context(task_dir) is None

    def test_returns_empty_dict_for_empty_shared(self, tmp_path: Path) -> None:
        """An empty shared.yaml returns an empty dict (yaml.safe_load → None → {})."""
        domain = _make_domain(tmp_path)
        subdir = domain / "tasks" / "subdir"
        subdir.mkdir(parents=True, exist_ok=True)
        (subdir / "shared.yaml").write_text("")
        loader = ConfigLoader(domain)
        # Empty YAML → None, but the method should normalize to {}
        result = loader._load_shared_context(subdir)
        assert result == {}


# ── _discover_task_files tests ──────────────────────────────────────


class TestDiscoverTaskFiles:
    """Tests for ConfigLoader._discover_task_files."""

    def test_finds_yaml_files_recursively(self, tmp_path: Path) -> None:
        """Finds task .yaml files in subdirectories."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"domain": "test"},
            tasks={
                "subdir/task_a.yaml": [_minimal_task("a")],
                "subdir/task_b.yaml": [_minimal_task("b")],
                "other/task_c.yaml": [_minimal_task("c")],
            },
        )
        loader = ConfigLoader(domain)
        files = loader._discover_task_files()
        names = sorted(p.name for p in files)
        assert names == ["task_a.yaml", "task_b.yaml", "task_c.yaml"]

    def test_excludes_global_yaml(self, tmp_path: Path) -> None:
        """global.yaml in tasks_dir is excluded."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"domain": "test"},
            tasks={"subdir/task_a.yaml": [_minimal_task("a")]},
        )
        loader = ConfigLoader(domain)
        files = loader._discover_task_files()
        assert all(p.name != "global.yaml" for p in files)

    def test_excludes_shared_yaml(self, tmp_path: Path) -> None:
        """shared.yaml files are excluded."""
        domain = _make_domain(
            tmp_path,
            shared_yamls={"subdir": {"key": "val"}},
            tasks={"subdir/task_a.yaml": [_minimal_task("a")]},
        )
        loader = ConfigLoader(domain)
        files = loader._discover_task_files()
        assert all(p.name != "shared.yaml" for p in files)

    def test_handles_empty_tasks_dir(self, tmp_path: Path) -> None:
        """Empty tasks dir returns an empty list."""
        domain = _make_domain(tmp_path)
        loader = ConfigLoader(domain)
        assert loader._discover_task_files() == []

    def test_handles_missing_tasks_dir(self, tmp_path: Path) -> None:
        """If tasks/ doesn't exist, return empty list."""
        loader = ConfigLoader(tmp_path)
        assert loader._discover_task_files() == []


# ── _load_task_file tests ───────────────────────────────────────────


class TestLoadTaskFile:
    """Tests for ConfigLoader._load_task_file."""

    def test_single_task(self, tmp_path: Path) -> None:
        """File with one task under 'tasks' key returns a single-element list."""
        domain = _make_domain(
            tmp_path,
            tasks={"subdir/one.yaml": [_minimal_task("t1")]},
        )
        loader = ConfigLoader(domain)
        result = loader._load_task_file(domain / "tasks" / "subdir" / "one.yaml")
        assert len(result) == 1
        assert result[0]["task_id"] == "t1"

    def test_multiple_tasks(self, tmp_path: Path) -> None:
        """File with multiple tasks returns all of them."""
        domain = _make_domain(
            tmp_path,
            tasks={
                "subdir/multi.yaml": [
                    _minimal_task("t1"),
                    _minimal_task("t2"),
                    _minimal_task("t3"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader._load_task_file(domain / "tasks" / "subdir" / "multi.yaml")
        assert len(result) == 3
        ids = [t["task_id"] for t in result]
        assert ids == ["t1", "t2", "t3"]

    def test_missing_tasks_key_returns_empty(self, tmp_path: Path) -> None:
        """File without a 'tasks' key returns empty list."""
        domain = _make_domain(tmp_path)
        task_file = domain / "tasks" / "subdir" / "bad.yaml"
        _write_yaml(task_file, {"not_tasks": [{"task_id": "x"}]})
        loader = ConfigLoader(domain)
        assert loader._load_task_file(task_file) == []

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        """An empty YAML file returns empty list."""
        domain = _make_domain(tmp_path)
        task_file = domain / "tasks" / "subdir" / "empty.yaml"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("")
        loader = ConfigLoader(domain)
        assert loader._load_task_file(task_file) == []

    def test_non_dict_task_entry_raises(self, tmp_path: Path) -> None:
        """A non-dict item in the tasks list raises ValueError."""
        domain = _make_domain(tmp_path)
        task_file = domain / "tasks" / "subdir" / "bad.yaml"
        _write_yaml(task_file, {"tasks": ["not_a_dict"]})
        loader = ConfigLoader(domain)
        with pytest.raises(ValueError, match="Non-dict task entry"):
            loader._load_task_file(task_file)


# ── _apply_filter tests ────────────────────────────────────────────


class TestApplyFilter:
    """Tests for ConfigLoader._apply_filter."""

    @pytest.fixture()
    def sample_tasks(self) -> list[TaskConfig]:
        """A set of TaskConfig objects with varied IDs for filtering."""
        prompts = {
            "instruction": "i.j2",
            "assistant": "a.j2",
            "submit": "s.j2",
            "continue": "c.j2",
        }
        ids = [
            "incident_5_task_1",
            "incident_5_task_2",
            "incident_34_task_1",
            "incident_34_task_2",
            "incident_99_task_1",
        ]
        return [
            TaskConfig(
                task_id=tid,
                title=tid,
                description=f"desc {tid}",
                prompts=prompts,
            )
            for tid in ids
        ]

    def test_glob_star(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """Glob pattern 'incident_5*' matches tasks starting with incident_5."""
        loader = ConfigLoader(tmp_path)
        result = loader._apply_filter(sample_tasks, "incident_5*")
        ids = [t.task_id for t in result]
        assert ids == ["incident_5_task_1", "incident_5_task_2"]

    def test_glob_middle_wildcard(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """Glob pattern 'incident_*_task_1' matches all task_1 variants."""
        loader = ConfigLoader(tmp_path)
        result = loader._apply_filter(sample_tasks, "incident_*_task_1")
        ids = [t.task_id for t in result]
        assert ids == ["incident_5_task_1", "incident_34_task_1", "incident_99_task_1"]

    def test_comma_separated_ids(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """Comma-separated exact IDs: 'incident_5_task_1,incident_99_task_1'."""
        loader = ConfigLoader(tmp_path)
        result = loader._apply_filter(sample_tasks, "incident_5_task_1,incident_99_task_1")
        ids = [t.task_id for t in result]
        assert ids == ["incident_5_task_1", "incident_99_task_1"]

    def test_mixed_glob_and_exact(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """Mixed: 'incident_5*,incident_34_task_2'."""
        loader = ConfigLoader(tmp_path)
        result = loader._apply_filter(sample_tasks, "incident_5*,incident_34_task_2")
        ids = [t.task_id for t in result]
        assert ids == [
            "incident_5_task_1",
            "incident_5_task_2",
            "incident_34_task_2",
        ]

    def test_no_match_returns_empty(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """A filter that matches nothing returns an empty list."""
        loader = ConfigLoader(tmp_path)
        result = loader._apply_filter(sample_tasks, "nonexistent*")
        assert result == []

    def test_none_pattern_returns_all(self, tmp_path: Path, sample_tasks: list[TaskConfig]) -> None:
        """None filter should not be called (load_tasks handles it), but if
        called directly, it must be tested at the load_tasks level."""
        # _apply_filter always receives a string; None is handled upstream.
        # We test the upstream behavior in TestLoadTasks instead.


# ── load_tasks integration tests ────────────────────────────────────


class TestLoadTasks:
    """Integration tests for ConfigLoader.load_tasks."""

    def test_full_pipeline(self, tmp_path: Path) -> None:
        """Full pipeline: global + shared + task → merged TaskConfig."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            shared_yamls={
                "subdir": {
                    "initial_context": {"db_host": "shared-db"},
                },
            },
            tasks={
                "subdir/task_a.yaml": [
                    _minimal_task(
                        "task_a",
                        inherit_shared=True,
                        initial_context={"question": "What is X?"},
                    ),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        task = tasks[0]
        assert isinstance(task, TaskConfig)
        assert task.task_id == "task_a"
        # Shared context merged into initial_context
        assert task.initial_context["db_host"] == "shared-db"
        # Task-level context also present
        assert task.initial_context["question"] == "What is X?"
        # Global prompts applied
        assert task.prompts.instruction == "instructions/demo.j2"

    def test_inherit_shared_false_skips_shared(self, tmp_path: Path) -> None:
        """Task with inherit_shared: false should NOT get shared context."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            shared_yamls={
                "subdir": {"initial_context": {"shared_key": "shared_val"}},
            },
            tasks={
                "subdir/task_a.yaml": [
                    _minimal_task(
                        "task_a",
                        inherit_shared=False,
                        initial_context={"own_key": "own_val"},
                    ),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert "shared_key" not in tasks[0].initial_context
        assert tasks[0].initial_context["own_key"] == "own_val"

    def test_missing_shared_still_works(self, tmp_path: Path) -> None:
        """Tasks work even when there's no shared.yaml in their directory."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            tasks={
                "subdir/task_a.yaml": [
                    _minimal_task("task_a"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "task_a"

    def test_empty_domain_returns_empty(self, tmp_path: Path) -> None:
        """A domain with no task files returns an empty list."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
        )
        loader = ConfigLoader(domain)
        assert loader.load_tasks() == []

    def test_multiple_tasks_in_one_file(self, tmp_path: Path) -> None:
        """A file with multiple tasks under 'tasks:' key produces N TaskConfig objects."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            tasks={
                "subdir/multi.yaml": [
                    _minimal_task("task_1"),
                    _minimal_task("task_2"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 2
        ids = sorted(t.task_id for t in tasks)
        assert ids == ["task_1", "task_2"]

    def test_multiple_files_across_subdirs(self, tmp_path: Path) -> None:
        """Tasks from different subdirectories are all discovered."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            tasks={
                "dir_a/task_a.yaml": [_minimal_task("a")],
                "dir_b/task_b.yaml": [_minimal_task("b")],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        ids = sorted(t.task_id for t in tasks)
        assert ids == ["a", "b"]

    def test_task_variables_substitute_judge_llm_in_llm_judge_model(self, tmp_path: Path) -> None:
        """``{judge_llm}`` placeholder is substituted before TaskConfig validation."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "scoring_defaults": {
                        "llm_judge": {
                            "judge_system_template": "judge/system.j2",
                            "judge_user_template": "judge/user.j2",
                        }
                    },
                }
            },
            tasks={
                "sub/task_with_llm.yaml": [
                    _minimal_task(
                        "llm_task_1",
                        scoring={
                            "llm_judge": {
                                "checkpoint_1": {
                                    "target": "trajectory",
                                    "model": "{judge_llm}",
                                }
                            }
                        },
                    )
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks(task_variables={"judge_llm": "openai/azure/gpt-4.1-mini"})
        assert len(tasks) == 1
        assert len(tasks[0].scorers) == 1
        assert tasks[0].scorers[0].strategy == "llm_judge"
        assert tasks[0].scorers[0].criteria.model == "openai/azure/gpt-4.1-mini"  # type: ignore[attr-defined]

    def test_task_variables_absent_leave_placeholder_literal(self, tmp_path: Path) -> None:
        """Without task_variables, placeholders remain unchanged."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "scoring_defaults": {
                        "llm_judge": {
                            "judge_system_template": "judge/system.j2",
                            "judge_user_template": "judge/user.j2",
                        }
                    },
                }
            },
            tasks={
                "sub/task_with_llm.yaml": [
                    _minimal_task(
                        "llm_task_2",
                        scoring={
                            "llm_judge": {
                                "checkpoint_1": {
                                    "target": "trajectory",
                                    "model": "{judge_llm}",
                                }
                            }
                        },
                    )
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].scorers[0].criteria.model == "{judge_llm}"  # type: ignore[attr-defined]

    def test_task_filter_applied(self, tmp_path: Path) -> None:
        """load_tasks with task_filter only returns matching tasks."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            tasks={
                "dir/task_alpha.yaml": [_minimal_task("alpha_1")],
                "dir/task_beta.yaml": [_minimal_task("beta_1")],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks(task_filter="alpha*")
        assert len(tasks) == 1
        assert tasks[0].task_id == "alpha_1"

    def test_task_filter_none_returns_all(self, tmp_path: Path) -> None:
        """load_tasks with no filter returns all tasks."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            tasks={
                "dir/a.yaml": [_minimal_task("a")],
                "dir/b.yaml": [_minimal_task("b")],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks(task_filter=None)
        assert len(tasks) == 2

    def test_task_overrides_global_defaults(self, tmp_path: Path) -> None:
        """Task-level values override global defaults."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "max_steps": 10,
                },
            },
            tasks={
                "dir/task.yaml": [
                    _minimal_task("t1", max_steps=50),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert tasks[0].max_steps == 50

    def test_shared_overrides_global_defaults(self, tmp_path: Path) -> None:
        """Shared context overrides global defaults (global → shared → task)."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "initial_context": {"from_global": True, "shared_key": "global"},
                },
            },
            shared_yamls={
                "dir": {"initial_context": {"shared_key": "shared"}},
            },
            tasks={
                "dir/task.yaml": [
                    _minimal_task("t1", inherit_shared=True),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        # shared overrides global
        assert tasks[0].initial_context["shared_key"] == "shared"
        # global key preserved
        assert tasks[0].initial_context["from_global"] is True


# ── Error cases ─────────────────────────────────────────────────────


class TestErrorCases:
    """Tests for error handling in ConfigLoader."""

    def test_malformed_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed YAML in a task file should raise an error."""
        domain = _make_domain(tmp_path)
        task_file = domain / "tasks" / "subdir" / "bad.yaml"
        task_file.parent.mkdir(parents=True, exist_ok=True)
        task_file.write_text("tasks:\n  - task_id: [unterminated")
        loader = ConfigLoader(domain)
        with pytest.raises(yaml.YAMLError):
            loader._load_task_file(task_file)

    def test_missing_required_fields_raises_validation_error(
        self,
        tmp_path: Path,
    ) -> None:
        """A task that's missing required fields (e.g. prompts) after merge
        should fail Pydantic validation."""
        domain = _make_domain(
            tmp_path,
            # No global defaults → no prompts supplied
            tasks={
                "dir/task.yaml": [
                    {
                        "task_id": "missing_prompts",
                        "title": "No prompts",
                        "description": "Missing prompts field",
                    },
                ],
            },
        )
        loader = ConfigLoader(domain)
        with pytest.raises(Exception):  # noqa: B017 - ValidationError from Pydantic
            loader.load_tasks()

    def test_malformed_global_yaml_raises(self, tmp_path: Path) -> None:
        """Malformed global.yaml should raise a YAML error."""
        domain = _make_domain(tmp_path)
        global_file = domain / "tasks" / "global.yaml"
        global_file.write_text(": bad: yaml: [")
        loader = ConfigLoader(domain)
        with pytest.raises(yaml.YAMLError):
            loader._load_global_defaults()


# ── ConfigLoader init tests ─────────────────────────────────────────


class TestConfigLoaderInit:
    """Tests for ConfigLoader initialization."""

    def test_sets_domain_root(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path)
        assert loader._domain_root == tmp_path

    def test_sets_tasks_dir(self, tmp_path: Path) -> None:
        loader = ConfigLoader(tmp_path)
        assert loader._tasks_dir == tmp_path / "tasks"


# ── Replace-key cascade tests ──────────────────────────────────────


class TestToolsReplaceCascade:
    """tools: in YAML uses replace semantics (not deep merge)."""

    def test_task_tools_replace_global_tools(self, tmp_path: Path) -> None:
        """Task-level tools entirely replace global-level tools."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "tools": {
                        "bash": {"timeout": 120},
                        "python": {"timeout": 60},
                    },
                },
            },
            tasks={
                "dir/task.yaml": [
                    _minimal_task(
                        "t1",
                        tools={"bash": {"timeout": 30}},
                    ),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        # Task declared only bash → python should be gone (replace, not merge)
        assert "bash" in tasks[0].tools
        assert tasks[0].tools["bash"].timeout == 30
        assert "python" not in tasks[0].tools

    def test_no_task_tools_inherits_global(self, tmp_path: Path) -> None:
        """No tools key in task YAML → inherits global tools."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "tools": {
                        "bash": {"timeout": 120},
                        "python": {"timeout": 60},
                    },
                },
            },
            tasks={
                "dir/task.yaml": [_minimal_task("t1")],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert "bash" in tasks[0].tools
        assert tasks[0].tools["bash"].timeout == 120
        assert "python" in tasks[0].tools

    def test_empty_task_tools_clears_global(self, tmp_path: Path) -> None:
        """Task with empty tools: {} → replaces global (empty dict wins)."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": {
                    **_minimal_global_defaults(),
                    "tools": {
                        "bash": {"timeout": 120},
                    },
                },
            },
            tasks={
                "dir/task.yaml": [
                    _minimal_task("t1", tools={}),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert tasks[0].tools == {}

    def test_shared_tools_replaced_by_task(self, tmp_path: Path) -> None:
        """Shared defines tools, task redefines → task wins entirely."""
        domain = _make_domain(
            tmp_path,
            global_yaml={
                "domain": "test",
                "global_defaults": _minimal_global_defaults(),
            },
            shared_yamls={
                "dir": {
                    "tools": {
                        "bash": {"timeout": 100},
                        "python": {"timeout": 50},
                    },
                },
            },
            tasks={
                "dir/task.yaml": [
                    _minimal_task("t1", tools={"python": {"timeout": 99}}),
                ],
            },
        )
        loader = ConfigLoader(domain)
        tasks = loader.load_tasks()
        assert len(tasks) == 1
        assert "python" in tasks[0].tools
        assert tasks[0].tools["python"].timeout == 99
        assert "bash" not in tasks[0].tools  # replaced, not merged


# ── Tests for load_domain_config ────────────────────────────────────


def _minimal_domain_data(**overrides: object) -> dict[str, object]:
    """Return a minimal valid domain config dict."""
    base: dict[str, object] = {
        "slug": "test_domain",
        "name": "Test Domain",
        "description": "A test domain",
    }
    base.update(overrides)
    return base


class TestLoadDomainConfig:
    """Tests for the ``load_domain_config`` function."""

    def test_loads_minimal_domain(self, tmp_path: Path) -> None:
        """A minimal eval.yaml with required fields returns a DomainConfig."""
        _write_yaml(tmp_path / "eval.yaml", _minimal_domain_data())
        config = load_domain_config(tmp_path)
        assert isinstance(config, DomainConfig)
        assert config.slug == "test_domain"
        assert config.name == "Test Domain"
        assert config.description == "A test domain"
        assert config.version == "1.0.0"
        assert config.images == {}
        assert config.tags == []

    def test_loads_all_fields(self, tmp_path: Path) -> None:
        """All optional fields are loaded correctly."""
        data = _minimal_domain_data(
            version="2.0.0",
            tags=["security", "ctf"],
            maintainer="alice",
            documentation="https://docs.example.com",
            repository="https://github.com/example/repo",
            images={
                "sandbox": {
                    "tag": "sandbox:latest",
                    "dockerfile": "docker/Dockerfile.sandbox",
                    "context": ".",
                    "labels": {"env": "test"},
                    "build_args": {"BASE": "python:3.11"},
                },
            },
        )
        _write_yaml(tmp_path / "eval.yaml", data)
        config = load_domain_config(tmp_path)
        assert config.version == "2.0.0"
        assert config.tags == ["security", "ctf"]
        assert config.maintainer == "alice"
        assert "sandbox" in config.images
        img = config.images["sandbox"]
        assert img.tag == "sandbox:latest"
        assert img.dockerfile == "docker/Dockerfile.sandbox"
        assert img.context == "."
        assert img.labels == {"env": "test"}
        assert img.build_args == {"BASE": "python:3.11"}

    def test_file_not_found(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when eval.yaml doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_domain_config(tmp_path)

    def test_invalid_yaml_content(self, tmp_path: Path) -> None:
        """Raises ValidationError when YAML content is invalid for DomainConfig."""
        from pydantic import ValidationError

        _write_yaml(tmp_path / "eval.yaml", {"bad_key": "bad_value"})
        with pytest.raises(ValidationError):
            load_domain_config(tmp_path)

    def test_empty_yaml_file(self, tmp_path: Path) -> None:
        """Raises ValidationError for an empty YAML file."""
        from pydantic import ValidationError

        (tmp_path / "eval.yaml").write_text("")
        with pytest.raises(ValidationError):
            load_domain_config(tmp_path)

    def test_yaml_with_only_comments(self, tmp_path: Path) -> None:
        """Raises ValidationError when YAML file contains only comments."""
        from pydantic import ValidationError

        (tmp_path / "eval.yaml").write_text("# just a comment\n")
        with pytest.raises(ValidationError):
            load_domain_config(tmp_path)

    def test_result_is_frozen(self, tmp_path: Path) -> None:
        """The returned DomainConfig is immutable (frozen)."""
        _write_yaml(tmp_path / "eval.yaml", _minimal_domain_data())
        config = load_domain_config(tmp_path)
        with pytest.raises(Exception):  # noqa: B017 — ValidationError from frozen
            config.slug = "changed"  # type: ignore[misc]


# ── Dataset filtering tests ────────────────────────────────────────


class TestDatasetFiltering:
    """Tests for dataset-based task filtering in ConfigLoader.load_tasks()."""

    def test_dataset_filter_returns_matching_tasks(self, tmp_path: Path) -> None:
        """Only tasks with matching dataset are returned."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                    _minimal_task("t3", dataset="ds_a"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="ds_a")
        assert [t.task_id for t in result] == ["t1", "t3"]

    def test_dataset_filter_excludes_untagged_tasks(self, tmp_path: Path) -> None:
        """Tasks without dataset field are excluded when filtering is active."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2"),  # no dataset
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="ds_a")
        assert [t.task_id for t in result] == ["t1"]

    def test_no_dataset_no_default_returns_all(self, tmp_path: Path) -> None:
        """Without dataset param or default_dataset, all tasks are returned."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks()
        assert len(result) == 2

    def test_default_dataset_used_when_no_explicit(self, tmp_path: Path) -> None:
        """default_dataset from global.yaml is used as fallback."""
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks()  # no explicit dataset
        assert [t.task_id for t in result] == ["t1"]

    def test_explicit_dataset_overrides_default(self, tmp_path: Path) -> None:
        """Explicit dataset param overrides default_dataset."""
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="ds_b")
        assert [t.task_id for t in result] == ["t2"]

    def test_dataset_and_task_filter_compose(self, tmp_path: Path) -> None:
        """Dataset filtering + task_filter both apply (dataset first)."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("linux_001", dataset="ds_a"),
                    _minimal_task("linux_002", dataset="ds_a"),
                    _minimal_task("aks_001", dataset="ds_a"),
                    _minimal_task("linux_003", dataset="ds_b"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="ds_a", task_filter="linux_*")
        assert [t.task_id for t in result] == ["linux_001", "linux_002"]

    def test_dataset_inherited_from_shared(self, tmp_path: Path) -> None:
        """dataset can be inherited via shared.yaml merge cascade."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            shared_yamls={"sub": {"dataset": "ds_a"}},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1"),  # inherits dataset from shared
                    _minimal_task("t2", dataset="ds_b"),  # overrides shared
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="ds_a")
        assert [t.task_id for t in result] == ["t1"]

    def test_no_matching_dataset_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent dataset returns empty list."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="nonexistent")
        assert result == []

    def test_default_dataset_stripped_from_merge_cascade(self, tmp_path: Path) -> None:
        """default_dataset is stripped from global_defaults before merge cascade.

        Even though TaskConfig ignores unknown fields today, the merge cascade
        should not propagate global-only meta-keys into task dicts.
        """
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        # Without explicit dataset, default_dataset should filter
        result = loader.load_tasks()
        assert [t.task_id for t in result] == ["t1"]
        # Verify no task carries a stray 'default_dataset' attribute
        for task in result:
            assert not hasattr(task, "default_dataset")

    def test_dataset_all_returns_everything(self, tmp_path: Path) -> None:
        """dataset='all' returns all tasks regardless of dataset field."""
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                    _minimal_task("t3"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="all")
        assert [t.task_id for t in result] == ["t1", "t2", "t3"]

    def test_dataset_all_overrides_default(self, tmp_path: Path) -> None:
        """dataset='all' overrides default_dataset from global.yaml."""
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="all")
        assert [t.task_id for t in result] == ["t1", "t2"]

    def test_dataset_all_with_task_filter(self, tmp_path: Path) -> None:
        """dataset='all' + task_filter still applies the task filter."""
        defaults = _minimal_global_defaults()
        defaults["default_dataset"] = "ds_a"
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": defaults},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("linux_001", dataset="ds_a"),
                    _minimal_task("linux_002", dataset="ds_b"),
                    _minimal_task("aks_001", dataset="ds_a"),
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="all", task_filter="linux_*")
        assert [t.task_id for t in result] == ["linux_001", "linux_002"]

    def test_dataset_all_without_default_returns_all(self, tmp_path: Path) -> None:
        """dataset='all' returns every task even without default_dataset."""
        domain = _make_domain(
            tmp_path,
            global_yaml={"global_defaults": _minimal_global_defaults()},
            tasks={
                "sub/tasks.yaml": [
                    _minimal_task("t1", dataset="ds_a"),
                    _minimal_task("t2", dataset="ds_b"),
                    _minimal_task("t3"),  # no dataset
                ],
            },
        )
        loader = ConfigLoader(domain)
        result = loader.load_tasks(dataset="all")
        assert len(result) == 3
        assert [t.task_id for t in result] == ["t1", "t2", "t3"]
