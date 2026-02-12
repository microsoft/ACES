"""
Write executor for GitHub Copilot CLI compatibility.

This executor implements the 'write' tool interface expected by the GitHub Copilot CLI agent.
It writes new files with specified content, matching the exact schema from the
@github/copilot v0.0.384 package.

Schema (verified against actual CLI source):
    path: string           - Full absolute path to file to write. File MUST NOT exist.
    file_text?: string     - The content of the file to be written.

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
from ...base import CreateParameters, ExecutionContext, ExecutorParameters, Parameter, ParameterType
from ...exceptions import SandboxExecutionError
from ...models import CreateExecutorConfig, ExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class WriteExecutor(DockerExecutor):
    """
    Docker-based write executor for writing new files.

    Implements the Copilot CLI 'write' tool interface for writing files
    with specified content. The file must not already exist.

    Features:
    - Write files with specified content
    - Automatic parent directory creation
    - Verification that file doesn't already exist
    - Safe content encoding handling
    """

    _executor_metadata = {
        "name": "write",
        "description": "Write a new file with specified content. The file MUST NOT already exist.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return CreateParameters

    @classmethod
    def get_default_config(cls) -> CreateExecutorConfig:
        """Get default configuration for Write executor."""
        return CreateExecutorConfig(timeout=30.0)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> WriteExecutor:
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
        """Initialize Write executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up Write executor parameters matching Copilot CLI schema."""
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="Full absolute path to file to create. File MUST NOT exist.",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="file_text",
                type=ParameterType.STRING,
                description="The content of the file to be created.",
                required=False,
                default="",
            )
        )

    def _build_command(self, params: CreateParameters) -> list[str]:
        """
        Build shell command for creating a file.

        This constructs a shell command that:
        - Checks if file already exists (fails if it does, unless overwrite=True)
        - Creates parent directories if needed
        - Writes content to the file using heredoc

        Args:
            params: Typed create parameters

        Returns:
            List of command arguments for Docker execution
        """
        # Use base64 encoding to safely pass content with any characters
        import base64

        encoded_content = base64.b64encode(params.file_text.encode("utf-8")).decode("ascii")

        script_parts = []

        # Check if file already exists (unless overwrite is allowed)
        if not params.overwrite:
            script_parts.append(f'if [ -e "{params.path}" ]; then')
            script_parts.append(f'  echo "Error: File already exists: {params.path}"')
            script_parts.append("  exit 1")
            script_parts.append("fi")

        # Create parent directories if needed
        script_parts.append(f'mkdir -p "$(dirname "{params.path}")"')

        # Write content using base64 decoding for safe content handling
        if params.file_text:
            script_parts.append(f'echo "{encoded_content}" | base64 -d > "{params.path}"')
        else:
            # Create empty file
            script_parts.append(f'touch "{params.path}"')

        # Verify file was created
        script_parts.append(f'if [ -f "{params.path}" ]; then')
        script_parts.append(f'  echo "Successfully created: {params.path}"')
        script_parts.append("else")
        script_parts.append(f'  echo "Error: Failed to create file: {params.path}"')
        script_parts.append("  exit 1")
        script_parts.append("fi")

        script = "\n".join(script_parts)
        return ["/bin/sh", "-c", script]

    async def execute(self, params: CreateParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute the create command in Docker container.

        Args:
            params: Strongly-typed create parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with creation status
        """
        try:
            log_operation_start(
                logger,
                "create_file",
                episode_id=context.episode_id,
                path=params.path,
                content_length=len(params.file_text),
            )

            timeout = int(self.get_timeout())

            # Execute the create command
            command_args = self._build_command(params)

            try:
                result = await self._execute_in_container(context.episode_id, command_args, timeout)

                if result.exit_code == 0:
                    log_operation_success(
                        logger,
                        "create_file",
                        episode_id=context.episode_id,
                        path=params.path,
                    )
                else:
                    log_operation_failure(
                        logger,
                        "create_file",
                        episode_id=context.episode_id,
                        path=params.path,
                        error=result.stderr or result.stdout or "Unknown error",
                    )

                return result

            except Exception as e:
                log_operation_failure(
                    logger,
                    "create_file",
                    episode_id=context.episode_id,
                    path=params.path,
                    error=str(e),
                )
                raise

        except SandboxExecutionError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in create executor",
                extra={"error": str(e)},
                exc_info=True,
            )
            return CommandResult.error_result(error=f"Create execution error: {str(e)}")
