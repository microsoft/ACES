"""Configuration loading and deep merge utilities for SABER's 3-level YAML inheritance cascade.

Provides pure-dict merge functions that operate BEFORE Pydantic validation,
and ``ConfigLoader`` for discovering and loading domain task YAML files.

The cascade is: global.yaml → shared.yaml → task.yaml.
"""

from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path

import yaml

from saber.config.models import DomainConfig, GlobalDefaults, TaskConfig

__all__ = ["ConfigLoader", "deep_merge", "load_domain_config", "merge_task_configs"]

# Sentinel dataset value that bypasses dataset filtering.
# When passed as ``dataset="all"`` to :meth:`ConfigLoader.load_tasks`,
# the filter step is skipped and every task is returned.
_DATASET_ALL: str = "all"

# Keys whose values are replaced entirely (not recursively merged)
# in the global → shared → task config cascade.
_REPLACE_KEYS: frozenset[str] = frozenset({"tools"})


def deep_merge(
    base: dict[str, object],
    override: dict[str, object],
    replace_keys: frozenset[str] = frozenset(),
) -> dict[str, object]:
    """Deep merge two configuration dicts. Override values win.

    Rules:
    - Scalars: override wins
    - Lists: override replaces (no append)
    - Dicts: recursive merge, override keys win
    - None values in override are skipped (don't erase base values)
    - Keys in *replace_keys* are replaced entirely (no recursive dict merge)

    Args:
        base: The base configuration dict (e.g., from global.yaml).
        override: The overriding configuration dict (e.g., from task.yaml).
        replace_keys: Keys whose values should be replaced entirely
            rather than recursively merged.

    Returns:
        New merged dict. Neither input is mutated.
    """
    result: dict[str, object] = {}

    # Start with all base keys
    for key, base_val in base.items():
        if key in override:
            override_val = override[key]
            if override_val is None:
                # None in override → keep base value
                result[key] = _deep_copy_value(base_val)
            elif key in replace_keys:
                # Replace key → override wins entirely, no recursive merge
                result[key] = _deep_copy_value(override_val)
            elif isinstance(base_val, dict) and isinstance(override_val, dict):
                # Both dicts → recursive merge
                result[key] = deep_merge(base_val, override_val, replace_keys)
            else:
                # Scalar, list, or type mismatch → override wins
                result[key] = _deep_copy_value(override_val)
        else:
            # Key only in base → preserve
            result[key] = _deep_copy_value(base_val)

    # Add keys only in override (skip None values)
    for key, override_val in override.items():
        if key not in base and override_val is not None:
            result[key] = _deep_copy_value(override_val)

    return result


def merge_task_configs(
    global_defaults: dict[str, object],
    shared_context: dict[str, object] | None,
    task_config: dict[str, object],
) -> dict[str, object]:
    """Merge the 3-level config inheritance cascade.

    Applies: global → shared → task, respecting ``inherit_shared: false``.

    Keys listed in ``_REPLACE_KEYS`` (e.g. ``tools``) are replaced entirely
    rather than recursively merged.

    Args:
        global_defaults: Global defaults from global.yaml.
        shared_context: Shared context from shared.yaml (may be None).
        task_config: Task-specific config from task.yaml.

    Returns:
        Fully merged config dict, ready for Pydantic validation.
        The ``inherit_shared`` key is consumed and removed from output.
    """
    inherit_shared = task_config.get("inherit_shared", True)

    # Build a task copy without the inherit_shared key
    task_without_flag: dict[str, object] = {k: v for k, v in task_config.items() if k != "inherit_shared"}

    # Start from global defaults
    result = deep_merge(global_defaults, {}, _REPLACE_KEYS)

    # Merge shared if allowed and present
    if inherit_shared is not False and shared_context is not None:
        result = deep_merge(result, shared_context, _REPLACE_KEYS)

    # Task overrides always apply last
    result = deep_merge(result, task_without_flag, _REPLACE_KEYS)

    # Defensive: ensure inherit_shared never leaks from global/shared
    result.pop("inherit_shared", None)

    return result


def _deep_copy_value(value: object) -> object:
    """Create an independent copy of a value to prevent mutation leaking."""
    if isinstance(value, dict):
        return {k: _deep_copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_deep_copy_value(item) for item in value]
    # Scalars (str, int, float, bool, None) are immutable — no copy needed
    return value


_EXCLUDED_FILENAMES = frozenset({"global.yaml", "shared.yaml"})


class ConfigLoader:
    """Loads domain YAML configs with global → shared → task inheritance.

    The loader discovers task YAML files under ``domain_root/tasks/``,
    applies the 3-level merge cascade (global → shared → task), and returns
    fully-validated ``TaskConfig`` objects.

    Args:
        domain_root: Path to the domain config directory, e.g.
            ``domains/excytin/server/config/``.
    """

    def __init__(self, domain_root: Path) -> None:
        self._domain_root = domain_root
        self._tasks_dir = domain_root / "tasks"

    # ── public ──────────────────────────────────────────────────────

    def load_global_config(self) -> GlobalDefaults:
        """Load domain-wide settings from ``global.yaml``.

        Reads the ``global_defaults`` section and returns a validated
        :class:`GlobalDefaults` model.  Fields not present in the YAML
        fall back to their model defaults.

        Returns:
            Validated :class:`GlobalDefaults` instance.
        """
        defaults = self._load_global_defaults()
        return GlobalDefaults(**defaults)

    def load_tasks(self, task_filter: str | None = None, dataset: str | None = None) -> list[TaskConfig]:
        """Load all tasks with inheritance applied.

        1. Load ``global.yaml`` from *tasks_dir*.
        2. Discover all task YAML files.
        3. For each task file, load its sibling ``shared.yaml`` if present.
        4. Merge: global → shared → task using :func:`merge_task_configs`.
        5. Validate each merged dict into :class:`TaskConfig`.
        6. Apply *dataset* filter if active (explicit param or ``default_dataset``).
        7. Apply *task_filter* if provided.

        Args:
            task_filter: Optional glob/comma-separated filter.
            dataset: Optional dataset name. When set, only tasks whose
                ``dataset`` field matches are returned. Falls back to
                ``default_dataset`` from ``global.yaml`` when ``None``.

        Returns:
            List of fully-resolved :class:`TaskConfig` objects.
        """
        global_defaults = self._load_global_defaults()

        # Extract default_dataset before merge cascade (not a TaskConfig field)
        default_dataset_value = global_defaults.pop("default_dataset", None)

        task_files = self._discover_task_files()

        configs: list[TaskConfig] = []
        for task_path in sorted(task_files):
            shared = self._load_shared_context(task_path.parent)
            raw_tasks = self._load_task_file(task_path)
            for raw in raw_tasks:
                merged = merge_task_configs(global_defaults, shared, raw)
                configs.append(TaskConfig(**merged))

        # Resolve effective dataset: explicit param > global default > None (all tasks)
        effective_dataset = dataset
        if effective_dataset is None and isinstance(default_dataset_value, str):
            effective_dataset = default_dataset_value

        if effective_dataset is not None and effective_dataset != _DATASET_ALL:
            configs = [t for t in configs if t.dataset == effective_dataset]

        if task_filter is not None:
            configs = self._apply_filter(configs, task_filter)

        return configs

    # ── private helpers ─────────────────────────────────────────────

    def _load_global_defaults(self) -> dict[str, object]:
        """Load ``global.yaml`` from *tasks_dir*.

        Returns:
            The value under the ``global_defaults`` key, or ``{}``.
        """
        path = self._tasks_dir / "global.yaml"
        if not path.exists():
            return {}
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return {}
        result = data.get("global_defaults")
        if not isinstance(result, dict):
            return {}
        return result

    def _load_shared_context(self, task_dir: Path) -> dict[str, object] | None:
        """Load ``shared.yaml`` from *task_dir* if it exists.

        Returns:
            Shared config dict, empty dict if file is empty, or ``None``
            if the file does not exist.
        """
        path = task_dir / "shared.yaml"
        if not path.exists():
            return None
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return {}
        return data

    def _discover_task_files(self) -> list[Path]:
        """Find all ``.yaml`` files under *tasks_dir*, excluding reserved names.

        Returns:
            Sorted list of paths to task YAML files.
        """
        if not self._tasks_dir.exists():
            return []
        return sorted(p for p in self._tasks_dir.rglob("*.yaml") if p.name not in _EXCLUDED_FILENAMES)

    def _load_task_file(self, path: Path) -> list[dict[str, object]]:
        """Load a task YAML file.

        Args:
            path: Path to the task YAML file.

        Returns:
            List of task dicts from the ``tasks`` key, or ``[]``
            if the key is absent or the file is empty.

        Raises:
            yaml.YAMLError: If the YAML is malformed.
        """
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict):
            return []
        tasks = data.get("tasks")
        if not isinstance(tasks, list):
            return []
        result: list[dict[str, object]] = []
        for item in tasks:
            if not isinstance(item, dict):
                msg = f"Non-dict task entry in {path}: {item!r}"
                raise ValueError(msg)
            result.append(item)
        return result

    def _apply_filter(self, tasks: list[TaskConfig], pattern: str) -> list[TaskConfig]:
        """Filter tasks by pattern.

        Supports:
        - Glob patterns: ``incident_5*``, ``incident_*_task_1``
        - Comma-separated IDs: ``task_1,task_2,task_3``
        - Mixed: ``incident_5*,incident_34_task_1``

        Args:
            tasks: List of :class:`TaskConfig` to filter.
            pattern: Comma-separated glob patterns.

        Returns:
            Filtered list preserving original order.
        """
        patterns = [p.strip() for p in pattern.split(",")]
        return [task for task in tasks if any(fnmatch(task.task_id, pat) for pat in patterns)]


def load_domain_config(domain_root: Path) -> DomainConfig:
    """Load and validate a domain configuration from ``eval.yaml``.

    Args:
        domain_root: Path to the domain root directory containing ``eval.yaml``.

    Returns:
        Validated :class:`DomainConfig` instance.

    Raises:
        FileNotFoundError: If ``eval.yaml`` does not exist in *domain_root*.
        pydantic.ValidationError: If the YAML content is invalid or empty.
    """
    eval_path = domain_root / "eval.yaml"
    if not eval_path.exists():
        msg = f"eval.yaml not found in {domain_root}"
        raise FileNotFoundError(msg)

    with eval_path.open() as fh:
        data = yaml.safe_load(fh)

    if data is None:
        data = {}

    return DomainConfig(**data)
