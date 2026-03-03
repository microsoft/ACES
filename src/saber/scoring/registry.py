"""Scoring strategy registry.

Maps strategy name strings to :class:`~saber.scoring.strategies.SaberScoringStrategy`
instances.  Pre-registers the built-in ``static`` and ``none`` strategies.
"""

from __future__ import annotations

import threading

from saber.scoring.strategies import (
    LLMJudgeStrategy,
    NoneStrategy,
    SaberScoringStrategy,
    StaticJaccardStrategy,
    StaticStrategy,
    ToolCallCountStrategy,
    ToolCallStrategy,
)


class ScoringStrategyRegistry:
    """Maps strategy names to SaberScoringStrategy instances."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._strategies: dict[str, SaberScoringStrategy] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register the built-in strategies."""
        self.register("static", StaticStrategy())
        self.register("none", NoneStrategy())
        self.register("static_jaccard", StaticJaccardStrategy())
        self.register("llm_judge", LLMJudgeStrategy())
        self.register("tool_call", ToolCallStrategy())
        self.register("tool_call_count", ToolCallCountStrategy())

    def register(self, name: str, strategy: SaberScoringStrategy, *, force: bool = False) -> None:
        """Register a strategy under *name*.

        Args:
            name: Strategy name.
            strategy: Strategy instance.
            force: If ``True``, overwrite an existing registration.

        Raises:
            ValueError: If *name* is already registered and *force* is ``False``.
        """
        with self._lock:
            if name in self._strategies and not force:
                raise ValueError(f"Strategy {name!r} is already registered. Available: {sorted(self._strategies)}")
            self._strategies[name] = strategy

    def get(self, name: str) -> SaberScoringStrategy:
        """Return the strategy for *name*, or raise :class:`KeyError`."""
        with self._lock:
            if name not in self._strategies:
                raise KeyError(f"Unknown scoring strategy: {name!r}. Available: {sorted(self._strategies)}")
            return self._strategies[name]

    def available(self) -> list[str]:
        """Return a sorted list of registered strategy names."""
        return sorted(self._strategies)
