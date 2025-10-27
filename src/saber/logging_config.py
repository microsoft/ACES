"""SABER standardized logging utilities."""

from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import MutableMapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, replace
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any, Iterator, Mapping, Optional

__all__ = [
    "LogCategory",
    "LoggingConfig",
    "SaberLogger",
    "init_logging",
    "get_saber_logger",
    "set_log_context",
    "update_log_context",
    "clear_log_context",
    "log_context",
    "log_session_start",
    "log_session_end",
    "log_operation_start",
    "log_operation_success",
    "log_operation_failure",
    "log_timeout",
    "get_session_manager_logger",
    "get_docker_logger",
    "get_execution_logger",
    "get_api_logger",
    "get_agent_logger",
    "get_cleanup_logger",
    "setup_file_logging",
]


class LogCategory(Enum):
    """Logical logging categories for SABER subsystems."""

    # Server Core Components
    SESSION_MANAGER = ("🏗️", "Session manager orchestration")
    REST_API = ("🌐", "REST API endpoints")
    MCP_API = ("🔌", "MCP API endpoints and tool execution")
    EVALUATION = ("📊", "Evaluation and benchmarking")

    # Execution Framework
    EXECUTION = ("⚙️", "Execution coordination")
    DOCKER = ("🐳", "Container and sandbox lifecycle")
    SECURITY = ("🔒", "Security enforcement and validation")
    TASK_EXEC = ("🏃", "Task execution pipeline")

    # Task & Episode Management
    TASK_MANAGER = ("📋", "Task configuration and management")
    EPISODE = ("🎬", "Episode lifecycle")
    POLICY = ("📝", "Policy documents and rules")

    # Client Side Components
    AGENT = ("🤖", "Client agent orchestration")
    HARNESS = ("🚀", "Client harness operations")
    COMMUNICATION = ("💬", "Client/server communication")
    LLM = ("🧠", "Language model operations")

    # System Operations
    CONFIG = ("🔧", "Configuration and startup")
    CLEANUP = ("🧹", "Resource cleanup and shutdown")
    WARNING = ("⚠️", "Warnings and degraded modes")
    ERROR = ("❌", "Error handling")

    # Specialized Operations
    NETWORK = ("🌐", "Network setup and connectivity")
    FILE_OPS = ("📁", "File operations and transfers")
    HEALTH = ("💚", "Health checks and monitoring")

    def __init__(self, emoji: str, description: str) -> None:
        self.emoji = emoji
        self.description = description


DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(levelname)-8s %(category)s %(category_emoji)s %(session)s%(message)s%(structured_text)s"
)
DEFAULT_JSON_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"
DEFAULT_TEXT_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LOG_FILE_NAME = "saber.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 10

_LOG_CONTEXT: ContextVar[dict[str, Any]] = ContextVar("saber_log_context", default={})
_LOGGING_LOCK = Lock()
_LOGGING_INITIALIZED = False
_ACTIVE_CONFIG: Optional["LoggingConfig"] = None

_RESERVED_EXTRA_KEYS = {
    "category",
    "category_emoji",
    "category_display",
    "module",
    "session",
    "structured_text",
    "_structured",
}
_DEPENDENCY_LOG_LEVELS: dict[str, int] = {
    "uvicorn": logging.WARNING,
    "uvicorn.access": logging.ERROR,
    "uvicorn.error": logging.ERROR,
    "httpx": logging.WARNING,
    "docker": logging.WARNING,
    "urllib3": logging.WARNING,
    "asyncio": logging.WARNING,
    "watchfiles": logging.ERROR,
}


@dataclass(frozen=True)
class LoggingConfig:
    """Concrete logging configuration derived from environment or code."""

    level: int = logging.INFO
    console: bool = True
    structured: bool = False
    enable_file: bool = True
    log_dir: Path = Path("logs")
    file_name: str = DEFAULT_LOG_FILE_NAME
    max_bytes: int = DEFAULT_MAX_BYTES
    backup_count: int = DEFAULT_BACKUP_COUNT

    @classmethod
    def from_env(cls) -> "LoggingConfig":
        """Build a logging configuration using SABER_* environment variables."""

        level_name = os.getenv("SABER_LOG_LEVEL", "INFO").strip() or "INFO"
        level = _resolve_log_level(level_name)

        format_mode = os.getenv("SABER_LOG_FORMAT", "text").strip().lower()
        if format_mode not in {"text", "json"}:
            raise ValueError("SABER_LOG_FORMAT must be either 'text' or 'json' (case-insensitive).")
        structured = format_mode == "json"

        enable_file = _resolve_bool_env("SABER_ENABLE_FILE_LOGGING", default=True)

        log_dir_env = _resolve_path_env("SABER_LOG_DIR") or _resolve_path_env("SABER_LOGS_DIRECTORY")
        log_dir = log_dir_env or Path("logs")

        file_name = os.getenv("SABER_LOG_FILE_NAME", DEFAULT_LOG_FILE_NAME).strip()
        if not file_name:
            raise ValueError("SABER_LOG_FILE_NAME cannot be empty when specified.")

        max_bytes = _resolve_int_env("SABER_LOG_MAX_BYTES", DEFAULT_MAX_BYTES, minimum=1024)
        backup_count = _resolve_int_env("SABER_LOG_BACKUP_COUNT", DEFAULT_BACKUP_COUNT, minimum=1)

        return cls(
            level=level,
            structured=structured,
            enable_file=enable_file,
            log_dir=log_dir,
            file_name=file_name,
            max_bytes=max_bytes,
            backup_count=backup_count,
        )


class SaberLogger(logging.LoggerAdapter):
    """Logger adapter enforcing SABER structured logging conventions."""

    def __init__(self, logger: logging.Logger, category: LogCategory, module: str) -> None:
        super().__init__(logger, {})
        self._category = category
        self._module = module

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        if not isinstance(msg, str):
            msg = str(msg)

        supplied_extra = kwargs.pop("extra", None)
        normalized_extra = self._normalize_extra(supplied_extra)

        context_values = dict(_LOG_CONTEXT.get())
        context_values.update(normalized_extra)

        session_identifier = context_values.get("session_id")

        metadata: dict[str, Any] = {
            "category": self._category.name,
            "category_display": self._category.description,
            "category_emoji": self._category.emoji,
            "session": f"[{session_identifier}] " if session_identifier else "",
        }

        kwargs["extra"] = {**metadata, "_structured": context_values}
        return msg, kwargs

    def _normalize_extra(self, extra: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        if extra is None:
            return {}
        if not isinstance(extra, Mapping):
            raise TypeError("Logger extra payload must be a mapping of key/value pairs.")
        for key in extra.keys():
            if key in _RESERVED_EXTRA_KEYS:
                raise ValueError(f"'{key}' is a reserved logging field; use a different key name.")
        return dict(extra)


class _SaberTextFormatter(logging.Formatter):
    """Human-readable formatter with structured payload support."""

    def __init__(self, fmt: str = DEFAULT_LOG_FORMAT, datefmt: str = DEFAULT_TEXT_DATEFMT) -> None:
        super().__init__(fmt=fmt, datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        structured = getattr(record, "_structured", {})
        if not hasattr(record, "category"):
            record.category = "UNCATEGORIZED"
        if not hasattr(record, "category_display"):
            record.category_display = ""
        if not hasattr(record, "category_emoji"):
            record.category_emoji = ""
        if not hasattr(record, "session"):
            record.session = ""
        record.structured_text = _format_structured_pairs(structured)
        try:
            return super().format(record)
        finally:
            if hasattr(record, "structured_text"):
                delattr(record, "structured_text")


class _SaberJsonFormatter(logging.Formatter):
    """Structured JSON formatter for machine consumption."""

    def __init__(self, datefmt: str = DEFAULT_JSON_DATEFMT) -> None:
        super().__init__(datefmt=datefmt)

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": getattr(record, "module", None),
            "category": getattr(record, "category", "UNCATEGORIZED"),
            "category_display": getattr(record, "category_display", ""),
            "category_emoji": getattr(record, "category_emoji", ""),
            "message": record.getMessage(),
        }

        structured = getattr(record, "_structured", {})
        for key, value in structured.items():
            payload[key] = _serialize_for_json(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def init_logging(config: Optional[LoggingConfig] = None, *, force: bool = False) -> LoggingConfig:
    """Bootstrap the SABER logging stack.

    Args:
        config: Explicit logging configuration (defaults to environment configuration).
        force: Force re-initialization even if logging is already configured.

    Returns:
        The active logging configuration after initialization.
    """

    global _LOGGING_INITIALIZED, _ACTIVE_CONFIG

    with _LOGGING_LOCK:
        if _LOGGING_INITIALIZED and not force:
            return _ACTIVE_CONFIG if _ACTIVE_CONFIG is not None else LoggingConfig.from_env()

        resolved_config = config or LoggingConfig.from_env()

        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        root_logger.setLevel(resolved_config.level)

        if resolved_config.console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(resolved_config.level)
            console_handler.setFormatter(_SaberJsonFormatter() if resolved_config.structured else _SaberTextFormatter())
            root_logger.addHandler(console_handler)

        if resolved_config.enable_file:
            handler = _build_file_handler(resolved_config)
            root_logger.addHandler(handler)

        _configure_dependency_loggers()

        _LOGGING_INITIALIZED = True
        _ACTIVE_CONFIG = resolved_config
        return resolved_config


def get_saber_logger(category: LogCategory, module: str) -> SaberLogger:
    """Create a SABER logger adapter for the given category and module."""

    base_logger = logging.getLogger(module)
    return SaberLogger(base_logger, category, module)


def set_log_context(**context: Any) -> None:
    """Replace the active structured logging context."""

    _LOG_CONTEXT.set(dict(context))


def update_log_context(**context: Any) -> Token[dict[str, Any]]:
    """Merge new context values into the current logging context."""

    current = dict(_LOG_CONTEXT.get())
    current.update(context)
    return _LOG_CONTEXT.set(current)


def clear_log_context() -> None:
    """Clear all structured logging context values."""

    _LOG_CONTEXT.set({})


@contextmanager
def log_context(**context: Any) -> Iterator[None]:
    """Context manager that temporarily augments the logging context."""

    token = update_log_context(**context)
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def log_session_start(logger: SaberLogger, session_id: str, details: str | None = None) -> None:
    """Emit a standardized session-start log event."""

    extra = {"event": "session_started", "session_id": session_id}
    if details:
        extra["details"] = details
    logger.info("Session started", extra=extra)


def log_session_end(logger: SaberLogger, session_id: str, reason: str | None = None) -> None:
    """Emit a standardized session-end log event."""

    extra = {"event": "session_ended", "session_id": session_id}
    if reason:
        extra["reason"] = reason
    logger.info("Session ended", extra=extra)


def log_operation_start(logger: SaberLogger, operation: str, session_id: Optional[str] = None, **kwargs: Any) -> None:
    """Log standardized operation start events."""

    extra = {"event": "operation_started", "operation": operation}
    if session_id:
        extra["session_id"] = session_id
    if kwargs:
        extra.update(kwargs)
    logger.info("Operation started", extra=extra)


def log_operation_success(logger: SaberLogger, operation: str, session_id: Optional[str] = None, **kwargs: Any) -> None:
    """Log standardized operation success events."""

    extra = {"event": "operation_completed", "operation": operation}
    if session_id:
        extra["session_id"] = session_id
    if kwargs:
        extra.update(kwargs)
    logger.info("Operation completed", extra=extra)


def log_operation_failure(
    logger: SaberLogger, operation: str, error: Any, session_id: Optional[str] = None, **kwargs: Any
) -> None:
    """Log standardized operation failure events."""

    extra = {
        "event": "operation_failed",
        "operation": operation,
        "error": str(error),
    }
    if session_id:
        extra["session_id"] = session_id
    if kwargs:
        extra.update(kwargs)

    # Check if this is a tool call limit error and use friendlier messaging
    error_str = str(error)
    if "tool call limit" in error_str.lower() or "LimitExceededError" in str(type(error).__name__):
        logger.info("Agent reached tool call limit", extra=extra)
    else:
        logger.error("Operation failed", extra=extra)


def log_timeout(
    logger: SaberLogger,
    operation: str,
    timeout: int,
    session_id: Optional[str] = None,
    **kwargs: Any,
) -> None:
    """Log standardized timeout events."""

    extra = {
        "event": "operation_timeout",
        "operation": operation,
        "timeout_seconds": timeout,
    }
    if session_id:
        extra["session_id"] = session_id
    if kwargs:
        extra.update(kwargs)
    logger.warning("Operation timed out", extra=extra)


def get_session_manager_logger(module: str) -> SaberLogger:
    """Logger helper for session management modules."""

    return get_saber_logger(LogCategory.SESSION_MANAGER, module)


def get_docker_logger(module: str) -> SaberLogger:
    """Logger helper for Docker integration modules."""

    return get_saber_logger(LogCategory.DOCKER, module)


def get_execution_logger(module: str) -> SaberLogger:
    """Logger helper for execution modules."""

    return get_saber_logger(LogCategory.EXECUTION, module)


def get_api_logger(module: str, is_mcp: bool = False) -> SaberLogger:
    """Logger helper for API modules."""

    category = LogCategory.MCP_API if is_mcp else LogCategory.REST_API
    return get_saber_logger(category, module)


def get_agent_logger(module: str) -> SaberLogger:
    """Logger helper for agent-related modules."""

    return get_saber_logger(LogCategory.AGENT, module)


def get_cleanup_logger(module: str) -> SaberLogger:
    """Logger helper for cleanup modules."""

    return get_saber_logger(LogCategory.CLEANUP, module)


def setup_file_logging(
    logs_directory: Optional[str] = None, enable_file_logging: Optional[bool] = None
) -> LoggingConfig:
    """Legacy helper retained for backwards compatibility with entry points."""

    warnings.warn(
        "setup_file_logging is deprecated; use init_logging() with LoggingConfig instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    config = LoggingConfig.from_env()

    if logs_directory is not None:
        config = replace(config, enable_file=True, log_dir=Path(logs_directory))

    if enable_file_logging is not None:
        config = replace(config, enable_file=_coerce_bool(enable_file_logging))

    return init_logging(config=config, force=True)


# Internal helpers ----------------------------------------------------------------


def _build_file_handler(config: LoggingConfig) -> logging.Handler:
    target_dir = config.log_dir.expanduser().resolve()
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise RuntimeError(f"Failed to create log directory '{target_dir}': {exc}") from exc

    log_path = target_dir / config.file_name
    handler = RotatingFileHandler(
        log_path,
        maxBytes=config.max_bytes,
        backupCount=config.backup_count,
        encoding="utf-8",
    )
    handler.setLevel(config.level)
    handler.setFormatter(_SaberJsonFormatter() if config.structured else _SaberTextFormatter())
    return handler


def _configure_dependency_loggers() -> None:
    for name, level in _DEPENDENCY_LOG_LEVELS.items():
        dep_logger = logging.getLogger(name)
        dep_logger.setLevel(level)
        dep_logger.propagate = True


def _resolve_log_level(level_name: str) -> int:
    level = logging.getLevelName(level_name.upper())
    if isinstance(level, str):
        raise ValueError(
            "Unsupported SABER_LOG_LEVEL '"
            f"{level_name}'. Use standard logging levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
        )
    return int(level)


def _resolve_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Environment variable {name} must be a boolean value, not '{raw}'.")


def _resolve_int_env(name: str, default: int, *, minimum: Optional[int] = None) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, not '{raw}'.") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"Environment variable {name} must be >= {minimum}, not {value}.")
    return value


def _resolve_path_env(name: str) -> Optional[Path]:
    raw = os.getenv(name)
    if not raw:
        return None
    return Path(raw).expanduser()


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"Boolean-like value expected, received {value!r}.")


def _format_structured_pairs(structured: Mapping[str, Any]) -> str:
    if not structured:
        return ""

    pairs = []
    for key in sorted(structured.keys()):
        if key == "session_id":
            continue
        value = structured[key]
        pairs.append(f"{key}={_stringify(value)}")
    return " | " + " ".join(pairs) if pairs else ""


def _stringify(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return str(value)
    return repr(value)


def _serialize_for_json(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _serialize_for_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_serialize_for_json(v) for v in value]
    return repr(value)
