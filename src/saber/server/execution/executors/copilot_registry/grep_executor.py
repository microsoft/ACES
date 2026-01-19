"""
Grep executor for GitHub Copilot CLI compatibility.

This executor implements the 'grep' tool interface expected by the GitHub Copilot CLI agent.
It provides fast and precise code search using regex patterns, matching the exact schema
from the @github/copilot v0.0.384 package.

Schema (verified against actual CLI source):
    pattern: string        - The regular expression pattern to search for in file contents
    path?: string          - File or directory to search in. Defaults to current working directory.
    glob?: string          - Glob pattern to filter files (e.g. "*.py")
    type?: string          - Optional file type filter (e.g., "py", "js")

The Copilot CLI uses ripgrep internally. This executor uses grep/find as fallback
since ripgrep may not be available in all sandbox environments.

Logging category: ``LogCategory.DOCKER``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...session_manager import SessionManager

from .....logging_config import (
    LogCategory,
    get_saber_logger,
    log_operation_failure,
    log_operation_start,
    log_operation_success,
)
from ....base import CommandResult
from ...base import ExecutionContext, ExecutorParameters, GrepParameters, Parameter, ParameterType
from ...exceptions import SandboxExecutionError
from ...models import ExecutorConfig, GrepExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)


# File type to extension mappings (matching ripgrep's type definitions)
FILE_TYPE_EXTENSIONS = {
    "py": ["*.py"],
    "python": ["*.py"],
    "js": ["*.js", "*.jsx", "*.mjs"],
    "javascript": ["*.js", "*.jsx", "*.mjs"],
    "ts": ["*.ts", "*.tsx"],
    "typescript": ["*.ts", "*.tsx"],
    "java": ["*.java"],
    "c": ["*.c", "*.h"],
    "cpp": ["*.cpp", "*.cc", "*.cxx", "*.hpp", "*.hh", "*.hxx", "*.h"],
    "go": ["*.go"],
    "rust": ["*.rs"],
    "ruby": ["*.rb"],
    "php": ["*.php"],
    "sh": ["*.sh", "*.bash"],
    "shell": ["*.sh", "*.bash"],
    "yaml": ["*.yaml", "*.yml"],
    "json": ["*.json"],
    "xml": ["*.xml"],
    "html": ["*.html", "*.htm"],
    "css": ["*.css"],
    "sql": ["*.sql"],
    "md": ["*.md", "*.markdown"],
    "markdown": ["*.md", "*.markdown"],
}


class GrepExecutor(DockerExecutor):
    """
    Docker-based grep executor for code search.

    Implements the Copilot CLI 'grep' tool interface for searching file contents
    using regular expressions. Tries to use ripgrep if available, falls back to grep.

    Features:
    - Regex pattern matching
    - Optional path restriction
    - Glob pattern filtering
    - File type filtering
    - Line number and file path output
    - Respects .gitignore by default
    """

    _executor_metadata = {
        "name": "grep",
        "description": "Search for patterns in file contents. Returns matching lines with file paths and line numbers.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return GrepParameters

    @classmethod
    def get_default_config(cls) -> GrepExecutorConfig:
        """Get default configuration for Grep executor."""
        return GrepExecutorConfig(timeout=60.0, max_results=500)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> GrepExecutor:
        """Create executor with standardized configuration interface."""
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(
            sandbox_manager=sandbox_manager,
            config=config,
            session_manager=session_manager,
            **merged_kwargs,
        )

    def __init__(
        self,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Grep executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up Grep executor parameters matching Copilot CLI schema."""
        self.add_parameter(
            Parameter(
                name="pattern",
                type=ParameterType.STRING,
                description="The regular expression pattern to search for in file contents.",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="File or directory to search in. Defaults to current working directory.",
                required=False,
                default=".",
            )
        )
        self.add_parameter(
            Parameter(
                name="glob",
                type=ParameterType.STRING,
                description="Glob pattern to filter files (e.g. '*.py').",
                required=False,
            )
        )
        self.add_parameter(
            Parameter(
                name="type",
                type=ParameterType.STRING,
                description="File type filter (e.g., 'py', 'js', 'ts').",
                required=False,
            )
        )

    def _get_glob_patterns_for_type(self, file_type: str) -> list[str]:
        """Get glob patterns for a file type."""
        return FILE_TYPE_EXTENSIONS.get(file_type.lower(), [f"*.{file_type}"])

    def _build_command(self, params: GrepParameters) -> list[str]:
        """
        Build shell command for grep search.

        This constructs a command that:
        - Tries ripgrep first (if available), falls back to grep
        - Applies glob and type filters
        - Returns results with file:line:content format
        - Limits output to prevent overwhelming results

        Args:
            params: Typed grep parameters

        Returns:
            List of command arguments for Docker execution
        """
        search_path = params.path or "."
        config = self._config
        max_results = config.max_results if isinstance(config, GrepExecutorConfig) else 500

        # Build glob patterns for filtering
        include_patterns = []
        if params.glob:
            include_patterns.append(params.glob)
        if params.file_type:
            include_patterns.extend(self._get_glob_patterns_for_type(params.file_type))

        # Escape pattern for shell
        escaped_pattern = params.pattern.replace("'", "'\"'\"'")

        script_parts = []

        # Try ripgrep first, fall back to grep
        script_parts.append("if command -v rg >/dev/null 2>&1; then")

        # Ripgrep command
        rg_cmd = "rg -n --color=never --no-heading"
        if include_patterns:
            for pat in include_patterns:
                rg_cmd += f" --glob '{pat}'"
        rg_cmd += f" -- '{escaped_pattern}' '{search_path}'"
        rg_cmd += f" | head -n {max_results}"

        script_parts.append(f"  {rg_cmd}")
        script_parts.append("else")

        # Fallback to grep with find
        if include_patterns:
            # Build find command with name filters
            name_args = " -o ".join([f"-name '{p}'" for p in include_patterns])
            find_cmd = f'find "{search_path}" -type f \\( {name_args} \\) 2>/dev/null'
        else:
            find_cmd = f'find "{search_path}" -type f 2>/dev/null'

        # Exclude hidden directories and common non-code directories
        find_cmd += ' | grep -v "/\\." | grep -v "/node_modules/" | grep -v "/__pycache__/"'

        grep_cmd = f"{find_cmd} | xargs grep -n -E '{escaped_pattern}' 2>/dev/null | head -n {max_results}"

        script_parts.append(f"  {grep_cmd}")
        script_parts.append("fi")

        # Handle no results gracefully
        script_parts.append("exit_code=$?")
        script_parts.append("if [ $exit_code -eq 1 ]; then")
        script_parts.append('  echo "No matches found"')
        script_parts.append("  exit 0")
        script_parts.append("fi")
        script_parts.append("exit $exit_code")

        script = "\n".join(script_parts)
        return ["/bin/sh", "-c", script]

    async def execute(self, params: GrepParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute the grep command in Docker container.

        Args:
            params: Strongly-typed grep parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with search results
        """
        try:
            log_operation_start(
                logger,
                "grep_search",
                episode_id=context.episode_id,
                pattern=params.pattern[:100],
                path=params.path,
                glob=params.glob,
                type=params.file_type,
            )

            # Get Docker environment for episode
            environment = self.get_episode_environment(context.episode_id)
            timeout = int(self.get_timeout())

            # Execute the grep command
            command_args = self._build_command(params)

            try:
                result = await environment.execute_command(command=command_args, timeout=timeout)

                # Count results
                output = result.stdout or ""
                result_count = len(output.strip().split("\n")) if output.strip() else 0

                if result.exit_code == 0:
                    log_operation_success(
                        logger,
                        "grep_search",
                        episode_id=context.episode_id,
                        pattern=params.pattern[:50],
                        result_count=result_count,
                    )
                else:
                    log_operation_failure(
                        logger,
                        "grep_search",
                        episode_id=context.episode_id,
                        pattern=params.pattern[:50],
                        error=result.stderr or "Unknown error",
                    )

                return result

            except Exception as e:
                log_operation_failure(
                    logger,
                    "grep_search",
                    episode_id=context.episode_id,
                    pattern=params.pattern[:50],
                    error=str(e),
                )
                raise

        except SandboxExecutionError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in grep executor",
                extra={"error": str(e)},
                exc_info=True,
            )
            return CommandResult.error_result(error=f"Grep execution error: {str(e)}")
