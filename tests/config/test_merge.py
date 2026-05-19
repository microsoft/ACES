# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""Tests for deep_merge and merge_task_configs utilities."""

from __future__ import annotations

import copy

import pytest

from saber.config.loader import deep_merge, merge_task_configs

# ── deep_merge tests ──────────────────────────────────────────────────────


class TestDeepMergeScalars:
    """Scalar override — child scalar replaces parent scalar."""

    def test_string_override(self) -> None:
        base = {"name": "parent"}
        override = {"name": "child"}
        result = deep_merge(base, override)
        assert result["name"] == "child"

    def test_int_override(self) -> None:
        base = {"count": 1}
        override = {"count": 42}
        assert deep_merge(base, override)["count"] == 42

    def test_bool_override(self) -> None:
        base = {"enabled": True}
        override = {"enabled": False}
        assert deep_merge(base, override)["enabled"] is False


class TestDeepMergeDicts:
    """Dict recursive merge — nested dicts are merged recursively."""

    def test_nested_dict_merge(self) -> None:
        base = {"server": {"host": "localhost", "port": 8080}}
        override = {"server": {"port": 9090}}
        result = deep_merge(base, override)
        assert result["server"] == {"host": "localhost", "port": 9090}

    def test_deep_nested_merge(self) -> None:
        """3+ levels deep dict merging."""
        base = {"a": {"b": {"c": {"d": 1, "e": 2}, "f": 3}}}
        override = {"a": {"b": {"c": {"d": 99}}}}
        result = deep_merge(base, override)
        assert result == {"a": {"b": {"c": {"d": 99, "e": 2}, "f": 3}}}


class TestDeepMergeLists:
    """List replacement — child list fully replaces parent list."""

    def test_list_replaced_not_appended(self) -> None:
        base = {"tags": ["a", "b", "c"]}
        override = {"tags": ["x"]}
        result = deep_merge(base, override)
        assert result["tags"] == ["x"]

    def test_empty_list_replaces(self) -> None:
        base = {"items": [1, 2, 3]}
        override = {"items": []}
        result = deep_merge(base, override)
        assert result["items"] == []


class TestDeepMergeNoneSkipping:
    """None values in override don't erase base values."""

    def test_none_does_not_override(self) -> None:
        base = {"key": "value"}
        override = {"key": None}
        result = deep_merge(base, override)
        assert result["key"] == "value"

    def test_none_in_nested_dict(self) -> None:
        base = {"server": {"host": "localhost", "port": 8080}}
        override = {"server": {"host": None, "port": 9090}}
        result = deep_merge(base, override)
        assert result["server"] == {"host": "localhost", "port": 9090}

    def test_false_is_not_treated_as_none(self) -> None:
        """False overrides base, None does not — explicitly contrasted."""
        base = {"flag": True, "name": "keep"}
        override = {"flag": False, "name": None}
        result = deep_merge(base, override)
        assert result["flag"] is False  # False overrides
        assert result["name"] == "keep"  # None is skipped

    def test_none_new_key_not_added(self) -> None:
        """A None value for a key not in base should not create the key."""
        result = deep_merge({"a": 1}, {"new_key": None})
        assert "new_key" not in result


class TestDeepMergeKeyPreservation:
    """Missing keys preserved and new keys added."""

    def test_base_only_keys_preserved(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"a": 10}
        result = deep_merge(base, override)
        assert result == {"a": 10, "b": 2}

    def test_override_only_keys_added(self) -> None:
        base = {"a": 1}
        override = {"b": 2}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 2}


class TestDeepMergeEmptyDicts:
    """Merging with empty base or empty override."""

    def test_empty_base(self) -> None:
        result = deep_merge({}, {"a": 1})
        assert result == {"a": 1}

    def test_empty_override(self) -> None:
        result = deep_merge({"a": 1}, {})
        assert result == {"a": 1}

    def test_both_empty(self) -> None:
        result = deep_merge({}, {})
        assert result == {}


class TestDeepMergeNoMutation:
    """Original dicts are not modified."""

    def test_base_not_mutated(self) -> None:
        base = {"server": {"host": "localhost", "port": 8080}}
        base_copy = copy.deepcopy(base)
        override = {"server": {"port": 9090}}
        deep_merge(base, override)
        assert base == base_copy

    def test_override_not_mutated(self) -> None:
        base = {"server": {"host": "localhost"}}
        override = {"server": {"port": 9090}}
        override_copy = copy.deepcopy(override)
        deep_merge(base, override)
        assert override == override_copy

    def test_result_is_independent(self) -> None:
        base = {"nested": {"key": "value"}}
        override = {"other": "data"}
        result = deep_merge(base, override)
        result["nested"]["key"] = "mutated"
        assert base["nested"]["key"] == "value"

    def test_nested_list_of_dicts_independent(self) -> None:
        """Lists containing dicts should be deep-copied, not shared."""
        base = {"items": [{"id": 1}, {"id": 2}]}
        result = deep_merge(base, {})
        result["items"][0]["id"] = 999
        assert base["items"][0]["id"] == 1  # base not mutated


class TestDeepMergeMixedTypes:
    """Scalar in base, dict in override → override wins (and vice versa)."""

    def test_scalar_to_dict(self) -> None:
        base = {"config": "simple_string"}
        override = {"config": {"nested": True}}
        result = deep_merge(base, override)
        assert result["config"] == {"nested": True}

    def test_dict_to_scalar(self) -> None:
        base = {"config": {"nested": True}}
        override = {"config": "simple_string"}
        result = deep_merge(base, override)
        assert result["config"] == "simple_string"

    def test_list_to_scalar(self) -> None:
        base = {"data": [1, 2, 3]}
        override = {"data": "replaced"}
        result = deep_merge(base, override)
        assert result["data"] == "replaced"


# ── merge_task_configs tests ──────────────────────────────────────────────


class TestMergeTaskConfigsCascade:
    """Full 3-level cascade and priority ordering."""

    def test_full_cascade(self) -> None:
        global_cfg = {"timeout": 30, "retries": 3, "server": {"host": "global"}}
        shared_cfg = {"retries": 5, "server": {"host": "shared", "port": 8080}}
        task_cfg = {"server": {"port": 9090}}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert result == {
            "timeout": 30,
            "retries": 5,
            "server": {"host": "shared", "port": 9090},
        }

    def test_task_wins_over_shared_wins_over_global(self) -> None:
        global_cfg = {"level": "global"}
        shared_cfg = {"level": "shared"}
        task_cfg = {"level": "task"}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert result["level"] == "task"


class TestMergeTaskConfigsInheritShared:
    """inherit_shared: false skips shared and is consumed."""

    def test_inherit_shared_false_skips_shared(self) -> None:
        global_cfg = {"a": 1}
        shared_cfg = {"a": 2, "b": "from_shared"}
        task_cfg = {"inherit_shared": False, "c": 3}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        # shared is skipped, so 'b' from shared is absent
        assert result == {"a": 1, "c": 3}

    def test_inherit_shared_consumed(self) -> None:
        global_cfg = {"x": 1}
        shared_cfg = {"y": 2}
        task_cfg = {"inherit_shared": False}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert "inherit_shared" not in result

    def test_inherit_shared_true_still_consumed(self) -> None:
        """Even inherit_shared: true should be consumed from the result."""
        global_cfg = {"x": 1}
        shared_cfg = {"y": 2}
        task_cfg = {"inherit_shared": True, "z": 3}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert "inherit_shared" not in result
        # shared IS merged when inherit_shared is true
        assert result == {"x": 1, "y": 2, "z": 3}

    def test_no_inherit_shared_key_means_inherit(self) -> None:
        """Without inherit_shared key, shared is merged (default behavior)."""
        global_cfg = {"a": 1}
        shared_cfg = {"b": 2}
        task_cfg = {"c": 3}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert result == {"a": 1, "b": 2, "c": 3}

    @pytest.mark.parametrize("value", [0, "", "false", None])
    def test_inherit_shared_only_skips_on_exact_false(self, value: object) -> None:
        """Only literal False should skip shared, not other falsy values."""
        result = merge_task_configs({"a": 1}, {"b": 2}, {"inherit_shared": value})
        assert "b" in result  # shared was NOT skipped
        assert "inherit_shared" not in result  # key consumed


class TestMergeTaskConfigsNoShared:
    """Works correctly when shared_context is None."""

    def test_none_shared(self) -> None:
        global_cfg = {"a": 1, "b": 2}
        task_cfg = {"b": 99, "c": 3}
        result = merge_task_configs(global_cfg, None, task_cfg)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_none_shared_with_inherit_shared_false(self) -> None:
        """inherit_shared: false with None shared still works and key is consumed."""
        global_cfg = {"a": 1}
        task_cfg = {"inherit_shared": False, "b": 2}
        result = merge_task_configs(global_cfg, None, task_cfg)
        assert result == {"a": 1, "b": 2}
        assert "inherit_shared" not in result


class TestMergeTaskConfigsNoMutation:
    """Inputs to merge_task_configs must not be mutated."""

    def test_merge_task_configs_does_not_mutate_inputs(self) -> None:
        """Verify that none of the three input dicts are mutated."""
        global_d = {"a": 1, "nested": {"x": 10}}
        shared_d = {"b": 2, "nested": {"y": 20}}
        task_d = {"c": 3, "nested": {"z": 30}}
        g_copy = copy.deepcopy(global_d)
        s_copy = copy.deepcopy(shared_d)
        t_copy = copy.deepcopy(task_d)
        merge_task_configs(global_d, shared_d, task_d)
        assert global_d == g_copy
        assert shared_d == s_copy
        assert task_d == t_copy


# ── deep_merge replace_keys tests ────────────────────────────────────────


class TestDeepMergeReplaceKeys:
    """Keys in replace_keys are replaced entirely, not recursively merged."""

    def test_replace_key_replaces_dict(self) -> None:
        """When 'tools' is a replace key, override replaces base entirely."""
        base = {"tools": {"bash": {"timeout": 180}, "python": {"timeout": 60}}}
        override = {"tools": {"nmap": {"timeout": 30}}}
        result = deep_merge(base, override, replace_keys=frozenset({"tools"}))
        assert result["tools"] == {"nmap": {"timeout": 30}}

    def test_non_replace_key_still_merges(self) -> None:
        """Keys NOT in replace_keys still get recursive merge."""
        base = {"config": {"a": 1, "b": 2}, "tools": {"bash": {}}}
        override = {"config": {"b": 99}, "tools": {"nmap": {}}}
        result = deep_merge(base, override, replace_keys=frozenset({"tools"}))
        assert result["config"] == {"a": 1, "b": 99}  # merged
        assert result["tools"] == {"nmap": {}}  # replaced

    def test_replace_key_absent_in_override_keeps_base(self) -> None:
        """When override doesn't have the replace key, base is kept."""
        base = {"tools": {"bash": {"timeout": 180}}}
        override = {"other": "val"}
        result = deep_merge(base, override, replace_keys=frozenset({"tools"}))
        assert result["tools"] == {"bash": {"timeout": 180}}

    def test_replace_key_none_in_override_keeps_base(self) -> None:
        """None in override for a replace key keeps the base value."""
        base = {"tools": {"bash": {}}}
        override = {"tools": None}
        result = deep_merge(base, override, replace_keys=frozenset({"tools"}))
        assert result["tools"] == {"bash": {}}

    def test_empty_replace_keys_behaves_as_before(self) -> None:
        """Empty replace_keys (default) gives old recursive merge behavior."""
        base = {"tools": {"bash": {"timeout": 180}}}
        override = {"tools": {"python": {"timeout": 60}}}
        result = deep_merge(base, override)
        assert result["tools"] == {"bash": {"timeout": 180}, "python": {"timeout": 60}}


# ── merge_task_configs tools replace semantics ───────────────────────────


class TestMergeTaskConfigsToolsReplace:
    """tools: key in merge_task_configs uses replace semantics."""

    def test_task_tools_replace_parent_tools(self) -> None:
        """When task defines tools:, it replaces parent tools entirely."""
        global_cfg = {"tools": {"bash": {"timeout": 180}, "python": {"timeout": 60}}}
        task_cfg = {"tools": {"nmap": {"timeout": 30}}}
        result = merge_task_configs(global_cfg, None, task_cfg)
        assert result["tools"] == {"nmap": {"timeout": 30}}

    def test_task_without_tools_inherits_parent(self) -> None:
        """When task does NOT define tools:, parent tools are inherited."""
        global_cfg = {"tools": {"bash": {"timeout": 180}}}
        task_cfg = {"other_key": "value"}
        result = merge_task_configs(global_cfg, None, task_cfg)
        assert result["tools"] == {"bash": {"timeout": 180}}

    def test_three_level_cascade_replace(self) -> None:
        """3-level cascade: global → shared → task replace semantics."""
        global_cfg = {"tools": {"bash": {"timeout": 180}}, "max_steps": 25}
        shared_cfg = {"tools": {"python": {"timeout": 60}}}
        task_cfg = {"tools": {"nmap": {"timeout": 30}}}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        # task tools replace shared tools (which replaced global tools)
        assert result["tools"] == {"nmap": {"timeout": 30}}
        # non-tools keys still cascade normally
        assert result["max_steps"] == 25

    def test_three_level_shared_replaces_global_task_inherits(self) -> None:
        """Shared replaces global tools, task without tools inherits shared."""
        global_cfg = {"tools": {"bash": {"timeout": 180}}}
        shared_cfg = {"tools": {"python": {"timeout": 60}}}
        task_cfg = {"max_steps": 50}
        result = merge_task_configs(global_cfg, shared_cfg, task_cfg)
        assert result["tools"] == {"python": {"timeout": 60}}
        assert result["max_steps"] == 50
