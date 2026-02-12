"""
Glob executor for GitHub Copilot CLI compatibility.

This executor implements the 'glob' tool interface expected by the GitHub Copilot CLI agent.
It provides fast file pattern matching using glob patterns, matching the exact schema
from the @github/copilot v0.0.384 package.

Schema (verified against actual CLI source):
    pattern: string        - Glob pattern to match files (e.g., "**/*.py", "src/*.ts")
    path?: string          - Directory to search in. Defaults to current working directory.

The Copilot CLI uses ripgrep internally with `rg --files --glob`. This executor
provides equivalent functionality using find/bash globbing as fallback since
ripgrep may not be available in all sandbox environments.

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
from ...base import ExecutionContext, ExecutorParameters, GlobParameters, Parameter, ParameterType
from ...exceptions import SandboxExecutionError
from ...models import ExecutorConfig, GlobExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class GlobExecutor(DockerExecutor):
    """
    Docker-based glob executor for file discovery.

    Implements the Copilot CLI 'glob' tool interface for finding files matching
    glob patterns. Tries to use ripgrep if available, falls back to find.

    Features:
    - Glob pattern matching (e.g., "**/*.py", "*.ts", "src/**/*")
    - Optional path restriction
    - Respects .gitignore by default (when using ripgrep)
    - Returns sorted list of matching file paths
    """

    _executor_metadata = {
        "name": "glob",
        "description": "Find files matching a glob pattern. Returns list of matching file paths.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return GlobParameters

    @classmethod
    def get_default_config(cls) -> GlobExecutorConfig:
        """Get default configuration for Glob executor."""
        return GlobExecutorConfig(timeout=30.0, max_results=1000)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> GlobExecutor:
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
        """Initialize Glob executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up Glob executor parameters matching Copilot CLI schema."""
        self.add_parameter(
            Parameter(
                name="pattern",
                type=ParameterType.STRING,
                description="Glob pattern to match files (e.g., '**/*.py', 'src/*.ts').",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="Directory to search in. Defaults to current working directory.",
                required=False,
                default=".",
            )
        )

    def _build_command(self, params: GlobParameters) -> list[str]:
        """
        Build shell command for glob file matching.

        This constructs a command that:
        - Tries ripgrep first (if available), falls back to find
        - Matches files against the glob pattern
        - Returns sorted list of file paths
        - Limits output to prevent overwhelming results

        Args:
            params: Typed glob parameters

        Returns:
            List of command arguments for Docker execution
        """
        search_path = params.path or "."
        config = self._config
        max_results = config.max_results if isinstance(config, GlobExecutorConfig) else 1000

        # Escape pattern for shell
        escaped_pattern = params.pattern.replace("'", "'\"'\"'")
        escaped_path = search_path.replace("'", "'\"'\"'")

        # Build the script that tries ripgrep first, then falls back to find
        script = f"""
# Try ripgrep first (faster, respects .gitignore)
if command -v rg >/dev/null 2>&1; then
    rg --files --glob '{escaped_pattern}' '{escaped_path}' 2>/dev/null | head -n {max_results} | sort
else
    # Fall back to find with glob pattern matching
    # Handle different glob patterns
    pattern='{escaped_pattern}'
    searchpath='{escaped_path}'

    # Check if pattern starts with **/ (recursive)
    if [[ "$pattern" == "**/"* ]]; then
        # Remove **/ prefix and search recursively
        subpattern="${{pattern#\\*\\*/}}"
        find "$searchpath" -type f -name "$subpattern" 2>/dev/null | head -n {max_results} | sort
    elif [[ "$pattern" == *"**"* ]]; then
        # Complex glob with ** in middle - use find with maxdepth
        # Extract the extension/name pattern after the last /
        name_pattern="${{pattern##*/}}"
        if [[ -n "$name_pattern" && "$name_pattern" != "$pattern" ]]; then
            find "$searchpath" -type f -name "$name_pattern" 2>/dev/null | head -n {max_results} | sort
        else
            find "$searchpath" -type f 2>/dev/null | head -n {max_results} | sort
        fi
    else
        # Simple glob pattern - use find -name
        find "$searchpath" -type f -name "$pattern" 2>/dev/null | head -n {max_results} | sort
    fi
fi
"""

        return ["bash", "-c", script.strip()]

    async def execute(self, params: GlobParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute the glob file search.

        Args:
            params: Strongly-typed glob parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with list of matching file paths
        """
        try:
            log_operation_start(
                logger,
                "glob_search",
                episode_id=context.episode_id,
                pattern=params.pattern,
                path=params.path,
            )

            timeout = int(self.get_timeout())

            # Execute the glob command
            command_args = self._build_command(params)

            try:
                result = await self._execute_in_container(context.episode_id, command_args, timeout)

                # Count results
                output = result.stdout or ""
                result_count = len(output.strip().split("\n")) if output.strip() else 0

                if result.exit_code == 0:
                    log_operation_success(
                        logger,
                        "glob_search",
                        episode_id=context.episode_id,
                        pattern=params.pattern,
                        result_count=result_count,
                    )
                else:
                    log_operation_failure(
                        logger,
                        "glob_search",
                        episode_id=context.episode_id,
                        pattern=params.pattern,
                        error=result.stderr or "Unknown error",
                    )

                return result

            except Exception as e:
                log_operation_failure(
                    logger,
                    "glob_search",
                    episode_id=context.episode_id,
                    pattern=params.pattern,
                    error=str(e),
                )
                raise

        except SandboxExecutionError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in glob executor",
                extra={"error": str(e)},
                exc_info=True,
            )
            return CommandResult.error_result(error=f"Glob execution error: {str(e)}")
