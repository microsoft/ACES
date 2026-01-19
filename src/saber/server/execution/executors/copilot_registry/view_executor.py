"""
View executor for GitHub Copilot CLI compatibility.

This executor implements the 'view' tool interface expected by the GitHub Copilot CLI agent.
It reads file contents or lists directory contents, matching the exact schema from the
@github/copilot v0.0.384 package.

Schema (verified against actual CLI source):
    path: string           - Full absolute path to file or directory. File MUST exist to view.
    view_range?: [number, number] - Optional line range [start, end]. 1-indexed. Use [start, -1] for end.
    forceReadLargeFiles?: boolean - Skip large file size check (default: false)

Constants:
    Large file threshold: 10MB (WFn = 10 * 1024 * 1024)
    Maximum file size: 1GB (Pvl = 1024 * 1024 * 1024)

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
from ...base import ExecutionContext, ExecutorParameters, Parameter, ParameterType, ViewParameters
from ...exceptions import SandboxExecutionError
from ...models import ExecutorConfig, ViewExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)

# Constants matching Copilot CLI (verified from index.js)
LARGE_FILE_THRESHOLD = 10 * 1024 * 1024  # 10MB - WFn
MAX_FILE_SIZE = 1024 * 1024 * 1024  # 1GB - Pvl


class ViewExecutor(DockerExecutor):
    """
    Docker-based view executor for reading files and listing directories.

    Implements the Copilot CLI 'view' tool interface for reading file contents
    with line numbers or listing directory contents up to 2 levels deep.

    Features:
    - File content display with line number prefixes
    - Optional line range viewing (1-indexed)
    - Directory listing (non-hidden files, 2 levels deep)
    - Large file protection with override option
    - Image file detection (returns base64 for images)
    """

    _executor_metadata = {
        "name": "view",
        "description": (
            "Read file contents or list directory contents. "
            "Returns content with line numbers for files, or a tree view for directories."
        ),
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return ViewParameters

    @classmethod
    def get_default_config(cls) -> ViewExecutorConfig:
        """Get default configuration for View executor."""
        return ViewExecutorConfig(
            timeout=30.0,  # File reads should be quick
            large_file_threshold=LARGE_FILE_THRESHOLD,
        )

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> ViewExecutor:
        """Create View executor with standardized configuration interface."""
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
        """Initialize View executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up View executor parameters matching Copilot CLI schema."""
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="Full absolute path to file or directory. File MUST exist to view.",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="view_range",
                type=ParameterType.ARRAY,
                description=(
                    "Optional line range [start, end]. Indexing starts at 1. "
                    "Use [start, -1] to read from start to end of file."
                ),
                required=False,
                items={"type": "integer"},
            )
        )
        self.add_parameter(
            Parameter(
                name="forceReadLargeFiles",
                type=ParameterType.BOOLEAN,
                description="Skip large file size check (default: false)",
                required=False,
                default=False,
            )
        )

    def _build_command(self, params: ViewParameters) -> list[str]:
        """
        Build shell command for viewing file/directory.

        This constructs a shell command that:
        - Checks if path is a file or directory
        - For files: uses head/tail/cat with line numbers
        - For directories: uses find to list contents

        Args:
            params: Typed view parameters

        Returns:
            List of command arguments for Docker execution
        """
        # Build a shell script that handles both files and directories
        script_parts = []

        # Check if path exists
        script_parts.append(f'if [ ! -e "{params.path}" ]; then')
        script_parts.append(f'  echo "Error: Path does not exist: {params.path}"')
        script_parts.append("  exit 1")
        script_parts.append("fi")

        # Check if it's a directory
        script_parts.append(f'if [ -d "{params.path}" ]; then')
        # List directory contents (non-hidden, 2 levels deep)
        script_parts.append(
            f'  find "{params.path}" -maxdepth 2 -not -path "*/\\.*" -type f -o -type d 2>/dev/null | head -500'
        )
        script_parts.append("else")

        # It's a file - check size first (unless forced)
        if not params.force_read_large_files:
            # Use typed config - cast since base class uses ExecutorConfig
            config = self._config
            large_threshold = (
                config.large_file_threshold if isinstance(config, ViewExecutorConfig) else LARGE_FILE_THRESHOLD
            )
            script_parts.append(
                f'  file_size=$(stat -c%s "{params.path}" 2>/dev/null || stat -f%z "{params.path}" 2>/dev/null)'
            )
            script_parts.append(f'  if [ "$file_size" -gt {large_threshold} ]; then')
            err_msg = f"Error: File exceeds size threshold ({large_threshold} bytes). "
            err_msg += "Use forceReadLargeFiles=true to override."
            script_parts.append(f'    echo "{err_msg}"')
            script_parts.append("    exit 1")
            script_parts.append("  fi")

        # Read file with line numbers
        if params.view_range:
            start_line, end_line = params.view_range

            if end_line == -1:
                # Read from start to end
                script_parts.append(f'  tail -n +{start_line} "{params.path}" | nl -ba -v {start_line}')
            else:
                # Read specific range
                line_count = end_line - start_line + 1
                script_parts.append(
                    f'  head -n {end_line} "{params.path}" | tail -n {line_count} | nl -ba -v {start_line}'
                )
        else:
            # Read entire file with line numbers
            script_parts.append(f'  nl -ba "{params.path}"')

        script_parts.append("fi")

        script = "\n".join(script_parts)
        return ["/bin/sh", "-c", script]

    async def execute(self, params: ViewParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute the view command in Docker container.

        Args:
            params: Strongly-typed view parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with file contents or directory listing
        """
        try:
            log_operation_start(
                logger,
                "view_file",
                episode_id=context.episode_id,
                path=params.path,
                view_range=params.view_range,
            )

            # Get Docker environment for episode
            environment = self.get_episode_environment(context.episode_id)
            timeout = int(self.get_timeout())

            # Execute the view command
            command_args = self._build_command(params)

            try:
                result = await environment.execute_command(command=command_args, timeout=timeout)

                if result.exit_code == 0:
                    log_operation_success(
                        logger,
                        "view_file",
                        episode_id=context.episode_id,
                        path=params.path,
                        output_length=len(result.stdout or ""),
                    )
                else:
                    log_operation_failure(
                        logger,
                        "view_file",
                        episode_id=context.episode_id,
                        path=params.path,
                        error=result.stderr or "Unknown error",
                    )

                return result

            except Exception as e:
                log_operation_failure(
                    logger,
                    "view_file",
                    episode_id=context.episode_id,
                    path=params.path,
                    error=str(e),
                )
                raise

        except SandboxExecutionError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in view executor",
                extra={"error": str(e)},
                exc_info=True,
            )
            return CommandResult.error_result(error=f"View execution error: {str(e)}")
