"""Evaluation system constants.

Logging Category: EVALUATION (no logging hooks defined in this module).
"""

# Import centralized constants from models layer
from ...models.constants import VALID_STEP_EVAL_STRATEGIES  # noqa: F401
from ...models.constants import VALID_SUBMISSION_EVAL_STRATEGIES  # noqa: F401
from ...models.constants import EvaluationStrategy  # noqa: F401
from ...models.constants import StepEvaluationStrategy  # noqa: F401
from ...models.constants import SubmissionEvaluationStrategy  # noqa: F401
from ...models.constants import EVAL_STRATEGY_LLM_JUDGE, EVAL_STRATEGY_STATIC, EVAL_STRATEGY_TOOL_CALL

# Backwards compatibility - kept for any code that might import from here
SUPPORTED_STRATEGIES = {EVAL_STRATEGY_STATIC, EVAL_STRATEGY_LLM_JUDGE, EVAL_STRATEGY_TOOL_CALL}
