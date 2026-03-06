"""Tests for saber.task — domain entry point helper."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saber.environments.images import RebuildMode, RebuildScope
from saber.task import _find_config_root, _find_prompts_dir, create_task


def _dummy_scorer():
    """Create a minimal valid inspect_ai Scorer for use in mocked tests."""
    from inspect_ai.scorer import Score, Scorer, Target, mean, scorer
    from inspect_ai.solver import TaskState

    @scorer(metrics=[mean()], name="test_dummy")  # type: ignore[misc]
    def dummy() -> Scorer:
        async def do_score(state: TaskState, target: Target) -> Score:
            return Score(value=0.0)

        return do_score

    return dummy()


# ── _find_config_root tests ──────────────────────────────────────────


class TestFindConfigRoot:
    def test_flat_layout(self, tmp_path: Path) -> None:
        """_find_config_root returns domain_root when tasks/ exists."""
        (tmp_path / "tasks").mkdir()
        result = _find_config_root(tmp_path)
        assert result == tmp_path

    def test_no_tasks_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when no tasks/ directory exists."""
        with pytest.raises(FileNotFoundError, match="No tasks/ directory found"):
            _find_config_root(tmp_path)

    def test_tasks_file_not_dir_not_found(self, tmp_path: Path) -> None:
        """A file named 'tasks' (not a directory) is not accepted."""
        (tmp_path / "tasks").write_text("not a dir")
        with pytest.raises(FileNotFoundError):
            _find_config_root(tmp_path)


# ── _find_prompts_dir tests ──────────────────────────────────────────


class TestFindPromptsDir:
    def test_flat_layout(self, tmp_path: Path) -> None:
        """_find_prompts_dir returns domain_root/prompts/ when it exists."""
        (tmp_path / "prompts").mkdir()
        result = _find_prompts_dir(tmp_path)
        assert result == tmp_path / "prompts"

    def test_no_prompts_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError when no prompts/ directory exists."""
        with pytest.raises(FileNotFoundError, match="No prompts/ directory found"):
            _find_prompts_dir(tmp_path)

    def test_prompts_file_not_dir_not_found(self, tmp_path: Path) -> None:
        """A file named 'prompts' (not a directory) is not accepted."""
        (tmp_path / "prompts").write_text("not a dir")
        with pytest.raises(FileNotFoundError):
            _find_prompts_dir(tmp_path)


# ── create_task tests ───────────────────────────────────────────────


def _write_minimal_domain(root: Path) -> None:
    """Create a minimal domain directory structure for testing."""
    # tasks/
    tasks_dir = root / "tasks"
    tasks_dir.mkdir()

    global_yaml = tasks_dir / "global.yaml"
    global_yaml.write_text(
        "global_defaults:\n"
        "  prompts:\n"
        '    instruction: "instructions/inst.md"\n'
        '    assistant: "assistants/asst.md"\n'
        '    submit: "submits/submit.md"\n'
        '    continue: "continues/continue.md"\n'
        "  max_steps: 5\n"
    )

    task_yaml = tasks_dir / "test_task.yaml"
    task_yaml.write_text(
        'tasks:\n  - task_id: test_task_1\n    title: "Test Task"\n    description: "A minimal test task"\n'
    )

    # prompts/
    prompts_dir = root / "prompts"
    for subdir, fname, content in [
        ("instructions", "inst.md", "You are a tester."),
        ("assistants", "asst.md", "Assistant prompt."),
        ("submits", "submit.md", "Submit your answer."),
        ("continues", "continue.md", "Continue working."),
    ]:
        d = prompts_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(content)


class TestCreateTask:
    def test_empty_domain_raises(self, tmp_path: Path) -> None:
        """Raises FileNotFoundError if no tasks/ directory exists."""
        (tmp_path / "prompts").mkdir()
        with pytest.raises(FileNotFoundError):
            create_task(tmp_path)

    def test_no_tasks_matching_filter_raises(self, tmp_path: Path) -> None:
        """Raises ValueError if filter matches no tasks."""
        _write_minimal_domain(tmp_path)
        with pytest.raises(ValueError, match="No tasks found"):
            # Use a mock agent since we just need to test the filter path
            with patch("saber.task.resolve_agent"):
                create_task(tmp_path, task_filter="nonexistent_*")

    def test_returns_task_with_valid_domain(self, tmp_path: Path) -> None:
        """create_task returns an inspect_ai Task with a valid domain."""
        _write_minimal_domain(tmp_path)

        # Mock the agent resolver so we don't need registered agents
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,  # skip permanent services
            )

        from inspect_ai import Task as InspectTask

        assert isinstance(task, InspectTask)

    def test_task_has_default_time_limit(self, tmp_path: Path) -> None:
        """create_task sets a default time_limit of 3600 on the Task."""
        _write_minimal_domain(tmp_path)

        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.time_limit == 3600

    def test_task_has_samples(self, tmp_path: Path) -> None:
        """The returned Task has the expected samples from YAML."""
        _write_minimal_domain(tmp_path)

        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.dataset is not None
        samples = list(task.dataset)
        assert len(samples) == 1
        assert samples[0].id == "test_task_1"

    def test_task_has_scorers(self, tmp_path: Path) -> None:
        """create_task wires scorers — returned task has scorer count > 0."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.scorer is not None
        # Each task gets at least an aggregate scorer
        assert len(task.scorer) >= 1  # type: ignore[arg-type]

    def test_permanent_compose_wired(self, tmp_path: Path) -> None:
        """create_task calls resolve_sandbox_spec with permanent compose args."""
        _write_minimal_domain(tmp_path)
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        perm_file = compose_dir / "databases.compose.yml"
        perm_file.write_text("version: '3'\nservices: {}")

        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", str(perm_file))) as mock_resolve,
        ):
            create_task(
                tmp_path,
                agent="react",
                permanent_compose="compose/databases.compose.yml",
            )

        mock_resolve.assert_called_once_with(
            domain_root=tmp_path,
            sandbox_compose="compose/sandbox.compose.yml",
            permanent_compose="compose/databases.compose.yml",
            permanent_project="saber-permanent",
            rebuild=RebuildMode.none(),
            keep_permanent=False,
        )

    def test_sandbox_spec_wired(self, tmp_path: Path) -> None:
        """create_task sets sandbox spec when compose file exists."""
        _write_minimal_domain(tmp_path)
        compose_dir = tmp_path / "compose"
        compose_dir.mkdir()
        sandbox_file = compose_dir / "sandbox.compose.yml"
        sandbox_file.write_text("version: '3'\nservices: {}")

        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.sandbox is not None
        # DEC-009: Task wraps ("saber", path) — always "saber" now
        sandbox_type = task.sandbox[0] if isinstance(task.sandbox, tuple) else task.sandbox.type
        assert sandbox_type == "saber"

    def test_tools_from_yaml_resolved(self, tmp_path: Path) -> None:
        """create_task resolves tools from YAML config and passes to solver."""
        _write_minimal_domain_with_tools(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.create_saber_solver") as mock_solver_factory,
        ):
            mock_solver_factory.return_value = MagicMock()
            create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        # Verify tool_registry was passed to create_saber_solver
        call_kwargs = mock_solver_factory.call_args
        assert "tool_registry" in call_kwargs.kwargs
        tool_registry = call_kwargs.kwargs["tool_registry"]
        assert tool_registry is not None
        # bash should be registered from YAML
        assert len(tool_registry.available()) >= 1

    def test_extra_tools_registered(self, tmp_path: Path) -> None:
        """create_task registers extra_tools from the domain @task function."""
        _write_minimal_domain_with_tools(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        def fake_tool() -> object:
            return object()

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.create_saber_solver") as mock_solver_factory,
        ):
            mock_solver_factory.return_value = MagicMock()
            create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                extra_tools={"my_custom": fake_tool},
            )

        call_kwargs = mock_solver_factory.call_args
        tool_registry = call_kwargs.kwargs.get("tool_registry")
        # bash from YAML + my_custom from extra_tools
        assert tool_registry is not None
        assert "my_custom" in tool_registry.available()


class TestExtraStrategies:
    """Tests for extra_strategies parameter on create_task."""

    def test_extra_strategies_passed_to_scorer_factory(self, tmp_path: Path) -> None:
        """When extra_strategies is provided, a custom registry is built and passed."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        fake_strategy = MagicMock()

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.ScorerFactory") as mock_scorer_cls,
        ):
            mock_scorer_cls.return_value.create_scorers.return_value = []
            mock_scorer_cls.return_value.create_overall_scorer.return_value = _dummy_scorer()
            create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                extra_strategies={"my_custom": fake_strategy},
            )

        call_kwargs = mock_scorer_cls.call_args.kwargs
        assert "registry" in call_kwargs
        registry = call_kwargs["registry"]

        from saber.scoring.registry import ScoringStrategyRegistry

        assert isinstance(registry, ScoringStrategyRegistry)
        # Should contain built-in defaults AND our custom strategy
        assert "my_custom" in registry.available()
        assert "static" in registry.available()  # built-in still present

    def test_extra_strategies_none_uses_defaults(self, tmp_path: Path) -> None:
        """When extra_strategies is None, ScorerFactory gets no custom registry."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.ScorerFactory") as mock_scorer_cls,
        ):
            mock_scorer_cls.return_value.create_scorers.return_value = []
            mock_scorer_cls.return_value.create_overall_scorer.return_value = _dummy_scorer()
            create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        call_kwargs = mock_scorer_cls.call_args.kwargs
        # registry should not be passed (or be None)
        assert call_kwargs.get("registry") is None

    def test_extra_strategies_available_in_registry(self, tmp_path: Path) -> None:
        """Integration: extra strategy is queryable from the real ScorerFactory registry."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        fake_strategy = MagicMock()

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                extra_strategies={"my_domain_strategy": fake_strategy},
            )

        # Task should be created successfully
        from inspect_ai import Task as InspectTask

        assert isinstance(task, InspectTask)


def _write_minimal_domain_with_tools(root: Path) -> None:
    """Create a minimal domain with tool configs in YAML."""
    tasks_dir = root / "tasks"
    tasks_dir.mkdir()

    global_yaml = tasks_dir / "global.yaml"
    global_yaml.write_text(
        "global_defaults:\n"
        "  prompts:\n"
        '    instruction: "instructions/inst.md"\n'
        '    assistant: "assistants/asst.md"\n'
        '    submit: "submits/submit.md"\n'
        '    continue: "continues/continue.md"\n'
        "  max_steps: 5\n"
        "  tools:\n"
        "    bash:\n"
        "      timeout: 120\n"
    )

    task_yaml = tasks_dir / "test_task.yaml"
    task_yaml.write_text(
        'tasks:\n  - task_id: test_task_1\n    title: "Test Task"\n    description: "A minimal test task"\n'
    )

    # prompts/
    prompts_dir = root / "prompts"
    for subdir, fname, content in [
        ("instructions", "inst.md", "You are a tester."),
        ("assistants", "asst.md", "Assistant prompt."),
        ("submits", "submit.md", "Submit your answer."),
        ("continues", "continue.md", "Continue working."),
    ]:
        d = prompts_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(content)


def _write_minimal_domain_with_approval(root: Path) -> None:
    """Create a minimal domain with approval config in YAML."""
    tasks_dir = root / "tasks"
    tasks_dir.mkdir()

    global_yaml = tasks_dir / "global.yaml"
    global_yaml.write_text(
        "global_defaults:\n"
        "  prompts:\n"
        '    instruction: "instructions/inst.md"\n'
        '    assistant: "assistants/asst.md"\n'
        '    submit: "submits/submit.md"\n'
        '    continue: "continues/continue.md"\n'
        "  max_steps: 5\n"
        "  approval:\n"
        "    tools:\n"
        '      - "bash"\n'
    )

    task_yaml = tasks_dir / "test_task.yaml"
    task_yaml.write_text(
        'tasks:\n  - task_id: test_task_1\n    title: "Test Task"\n    description: "A minimal test task"\n'
    )

    # prompts/
    prompts_dir = root / "prompts"
    for subdir, fname, content in [
        ("instructions", "inst.md", "You are a tester."),
        ("assistants", "asst.md", "Assistant prompt."),
        ("submits", "submit.md", "Submit your answer."),
        ("continues", "continue.md", "Continue working."),
    ]:
        d = prompts_dir / subdir
        d.mkdir(parents=True, exist_ok=True)
        (d / fname).write_text(content)


# ── Approval wiring tests ──────────────────────────────────────────


def _has_approval_module() -> bool:
    """Check if saber.approval module is available."""
    try:
        import saber.approval  # noqa: F401

        return True
    except ModuleNotFoundError:
        return False


class TestApprovalWiring:
    """Tests for approval parameter wiring in create_task."""

    def test_no_approval_no_task_approval(self, tmp_path: Path) -> None:
        """Default create_task with no approval → Task has no approval."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.approval is None

    @pytest.mark.skipif(
        not _has_approval_module(),
        reason="saber.approval module not yet implemented",
    )
    def test_explicit_approval_config_wired(self, tmp_path: Path) -> None:
        """create_task(approval=ApprovalConfig()) → Task has approval."""
        from saber.approval.models import ApprovalConfig

        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                approval=ApprovalConfig(),
            )

        assert task.approval is not None
        assert len(task.approval) >= 1

    @pytest.mark.skipif(
        not _has_approval_module(),
        reason="saber.approval module not yet implemented",
    )
    def test_yaml_approval_wired(self, tmp_path: Path) -> None:
        """YAML has approval config → Task has approval set."""
        _write_minimal_domain_with_approval(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
            )

        assert task.approval is not None
        assert len(task.approval) >= 1

    @pytest.mark.skipif(
        not _has_approval_module(),
        reason="saber.approval module not yet implemented",
    )
    def test_explicit_approval_overrides_yaml(self, tmp_path: Path) -> None:
        """Both YAML and param have approval → param wins."""
        from saber.approval.models import ApprovalConfig

        _write_minimal_domain_with_approval(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        explicit_config = ApprovalConfig(tools=("python", "exec"))

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                approval=explicit_config,
            )

        assert task.approval is not None
        assert len(task.approval) >= 1
        # Verify the explicit config was used (tools should be ["python", "exec"])
        policy = task.approval[0]
        from inspect_ai.approval import ApprovalPolicy

        assert isinstance(policy, ApprovalPolicy)
        assert policy.tools == ["python", "exec"]

    @pytest.mark.skipif(
        not _has_approval_module(),
        reason="saber.approval module not yet implemented",
    )
    def test_prebuilt_policies_passed_through(self, tmp_path: Path) -> None:
        """create_task(approval=[ApprovalPolicy(...)]) → passed as-is."""
        from inspect_ai.approval import ApprovalPolicy

        from saber.approval.approver import saber_security

        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        prebuilt = [ApprovalPolicy(approver=saber_security(), tools="*")]

        with patch("saber.task.resolve_agent", return_value=mock_factory):
            task = create_task(
                tmp_path,
                agent="react",
                permanent_compose=None,
                approval=prebuilt,
            )

        assert task.approval is prebuilt


# ── _resolve_approval tests ────────────────────────────────────────


@pytest.mark.skipif(
    not _has_approval_module(),
    reason="saber.approval module not yet implemented",
)
class TestResolveApproval:
    """Tests for _resolve_approval helper."""

    def test_none_returns_none(self) -> None:
        from saber.task import _resolve_approval

        assert _resolve_approval(None, []) is None

    def test_prebuilt_list_returned_as_is(self) -> None:
        from inspect_ai.approval import ApprovalPolicy

        from saber.approval.approver import saber_security
        from saber.task import _resolve_approval

        policies = [ApprovalPolicy(approver=saber_security(), tools="*")]
        result = _resolve_approval(policies, [])
        assert result is policies

    def test_explicit_config_builds_policies(self) -> None:
        from saber.approval.models import ApprovalConfig
        from saber.task import _resolve_approval

        config = ApprovalConfig(tools=("bash",))
        result = _resolve_approval(config, [])
        assert result is not None
        assert len(result) >= 1

    def test_task_config_approval_used_as_fallback(self) -> None:
        from saber.approval.models import ApprovalConfig
        from saber.config.models import PromptPaths, TaskConfig
        from saber.task import _resolve_approval

        tc = TaskConfig(
            task_id="t1",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            approval=ApprovalConfig(tools=("exec",)),
        )
        result = _resolve_approval(None, [tc])
        assert result is not None
        assert len(result) >= 1

    def test_empty_list_falls_through_to_yaml(self) -> None:
        """Empty list should not short-circuit — fall through to YAML config."""
        from saber.approval.models import ApprovalConfig
        from saber.config.models import PromptPaths, TaskConfig
        from saber.task import _resolve_approval

        tc = TaskConfig(
            task_id="t1",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            approval=ApprovalConfig(),
        )
        result = _resolve_approval([], [tc])
        assert result is not None
        assert len(result) > 0  # Should build from YAML config, not return []

    def test_explicit_config_beats_task_config(self) -> None:
        from saber.approval.models import ApprovalConfig
        from saber.config.models import PromptPaths, TaskConfig
        from saber.task import _resolve_approval

        explicit = ApprovalConfig(tools=("bash",))
        tc = TaskConfig(
            task_id="t1",
            title="T",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            approval=ApprovalConfig(tools=("exec",)),
        )
        result = _resolve_approval(explicit, [tc])
        assert result is not None
        # Should use explicit, not task config
        policy = result[0]
        assert policy.tools == ["bash"]

    def test_warns_on_multiple_different_approval_configs(self, caplog: pytest.LogCaptureFixture) -> None:
        """S5: Should log a warning when tasks have different approval configs."""
        from saber.approval.models import ApprovalConfig
        from saber.config.models import PromptPaths, TaskConfig
        from saber.task import _resolve_approval

        tc1 = TaskConfig(
            task_id="t1",
            title="T1",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            approval=ApprovalConfig(tools=("bash",)),
        )
        tc2 = TaskConfig(
            task_id="t2",
            title="T2",
            description="D",
            prompts=PromptPaths(instruction="i.j2"),
            approval=ApprovalConfig(tools=("exec",)),
        )
        # Ensure propagation is enabled so caplog (root handler) sees the
        # record.  configure_logging() in other tests may have set
        # propagate=False on the "saber" parent logger.
        saber_logger = logging.getLogger("saber")
        original_propagate = saber_logger.propagate
        saber_logger.propagate = True
        try:
            with caplog.at_level("WARNING", logger="saber.task"):
                result = _resolve_approval(None, [tc1, tc2])
        finally:
            saber_logger.propagate = original_propagate
        assert result is not None
        assert any("different approval" in r.message.lower() for r in caplog.records)


class TestCreateTaskRebuild:
    """Tests for rebuild parameter on create_task."""

    def test_rebuild_passed_through(self, tmp_path: Path) -> None:
        """create_task passes rebuild through to resolve_sandbox_spec."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")) as mock_resolve,
        ):
            create_task(tmp_path, agent="react", rebuild="true")

        call_kwargs = mock_resolve.call_args.kwargs
        assert "rebuild" in call_kwargs
        assert isinstance(call_kwargs["rebuild"], RebuildMode)
        assert call_kwargs["rebuild"].scope is RebuildScope.ALL

    def test_rebuild_none_default(self, tmp_path: Path) -> None:
        """When rebuild is None (default), passes RebuildMode.none()."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")) as mock_resolve,
        ):
            create_task(tmp_path, agent="react")

        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["rebuild"].scope is RebuildScope.NONE

    def test_rebuild_specific(self, tmp_path: Path) -> None:
        """Specific image names parsed and passed through."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")) as mock_resolve,
        ):
            create_task(tmp_path, agent="react", rebuild="web,db")

        call_kwargs = mock_resolve.call_args.kwargs
        assert call_kwargs["rebuild"].scope is RebuildScope.SPECIFIC
        assert call_kwargs["rebuild"].names == frozenset({"web", "db"})


class TestCreateTaskPreflight:
    """Tests for create_task run_preflight parameter."""

    def test_preflight_disabled_by_default(self, tmp_path: Path) -> None:
        """run_preflight=False (default) → validate_domain not called."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")),
            patch("saber.task.validate_domain") as mock_validate,
        ):
            create_task(tmp_path, agent="react")

        mock_validate.assert_not_called()

    def test_preflight_enabled_calls_validate_domain(self, tmp_path: Path) -> None:
        """run_preflight=True → validate_domain called."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        mock_result = MagicMock()
        mock_result.passed = True
        mock_result.warnings = ()

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")),
            patch("saber.task.validate_domain", return_value=mock_result) as mock_validate,
        ):
            create_task(tmp_path, agent="react", run_preflight=True)

        mock_validate.assert_called_once()

    def test_preflight_errors_raise_runtime_error(self, tmp_path: Path) -> None:
        """run_preflight=True with errors → RuntimeError."""
        _write_minimal_domain(tmp_path)

        from saber.environments.preflight import ComposePreflightResult, PreflightFinding

        error = PreflightFinding(path="compose/sandbox.compose.yml", message="Missing services", severity="error")
        result = ComposePreflightResult(findings=(error,))

        with (
            patch("saber.task.validate_domain", return_value=result),
            pytest.raises(RuntimeError, match="Compose preflight failed"),
        ):
            create_task(tmp_path, agent="react", run_preflight=True)

    def test_preflight_warnings_only_continues(self, tmp_path: Path) -> None:
        """run_preflight=True with only warnings → no exception, task created."""
        _write_minimal_domain(tmp_path)
        mock_factory = lambda: lambda **kwargs: lambda state, gen: state  # noqa: E731

        from saber.environments.preflight import ComposePreflightResult, PreflightFinding

        warning = PreflightFinding(path="compose/sandbox.compose.yml", message="Missing pids", severity="warning")
        result = ComposePreflightResult(findings=(warning,))

        with (
            patch("saber.task.resolve_agent", return_value=mock_factory),
            patch("saber.task.resolve_sandbox_spec", return_value=("saber", "/path")),
            patch("saber.task.validate_domain", return_value=result),
        ):
            task = create_task(tmp_path, agent="react", run_preflight=True)

        assert task is not None
