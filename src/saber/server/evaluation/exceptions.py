"""
Custom exceptions for evaluation system.
"""


class EvaluationError(Exception):
    """Base exception for evaluation errors."""

    pass


class EvaluationConfigError(EvaluationError):
    """Configuration-related evaluation error."""

    pass


class EvaluatorNotFoundError(EvaluationError):
    """Requested evaluator not available."""

    pass


class EvaluationValidationError(EvaluationError):
    """Error in evaluation validation."""

    pass


class InvalidEvaluationStrategyError(EvaluationError):
    """Invalid evaluation strategy specified."""

    pass


class MissingSubmissionError(EvaluationError):
    """Episode missing required submission for evaluation."""

    pass


class IncompleteEpisodeError(EvaluationError):
    """Cannot evaluate incomplete episode."""

    pass


class EvaluationRetrievalError(EvaluationError):
    """Base exception for evaluation retrieval errors."""

    pass


class EvaluationNotFoundError(EvaluationRetrievalError):
    """Evaluation result not found."""

    pass


class SessionEvaluationError(EvaluationRetrievalError):
    """Session-level evaluation access error."""

    pass


class InvalidEvaluationRequestError(EvaluationRetrievalError):
    """Invalid evaluation retrieval request parameters."""

    pass
