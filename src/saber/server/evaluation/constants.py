"""Evaluation system constants.

Logging Category: EVALUATION (no logging hooks defined in this module).
"""

# Evaluation strategy constants
EVAL_STRATEGY_STATIC = "static"
EVAL_STRATEGY_LLM_JUDGE = "llm_judge"

# Phase 2: Both static and LLM judge evaluation strategies are now supported
SUPPORTED_STRATEGIES = {EVAL_STRATEGY_STATIC, EVAL_STRATEGY_LLM_JUDGE}
