"""
Edit executor for GitHub Copilot CLI compatibility.

This executor implements the 'edit' (str_replace) tool interface expected by the
GitHub Copilot CLI agent. It makes string replacements in files, matching the exact
schema from the @github/copilot v0.0.384 package.

Schema (verified against actual CLI source):
    path: string           - Full absolute path to file to edit. File MUST exist.
    old_str: string        - The string in the file to replace. Preserve leading/trailing whitespace!
    new_str?: string       - The new string to replace old_str with.

Behavior:
    - Replaces exactly ONE occurrence of old_str with new_str
    - old_str must match EXACTLY (including whitespace)
    - If old_str is not unique, replacement will NOT be performed
    - If old_str is not found, returns error

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
from ...base import EditParameters, ExecutionContext, ExecutorParameters, Parameter, ParameterType
from ...exceptions import SandboxExecutionError
from ...models import EditExecutorConfig, ExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class EditExecutor(DockerExecutor):
    """
    Docker-based edit executor for string replacement in files.

    Implements the Copilot CLI 'edit' (str_replace) tool interface for making
    precise string replacements in files. Ensures exactly one occurrence is replaced.

    Features:
    - Exact string matching (including whitespace)
    - Single occurrence replacement only
    - Verification of uniqueness before replacement
    - Safe handling of special characters in strings
    """

    _executor_metadata = {
        "name": "edit",
        "description": "Make a string replacement in a file. Replaces exactly ONE occurrence of old_str with new_str.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return EditParameters

    @classmethod
    def get_default_config(cls) -> EditExecutorConfig:
        """Get default configuration for Edit executor."""
        return EditExecutorConfig(timeout=30.0)

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> EditExecutor:
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
        """Initialize Edit executor."""
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up Edit executor parameters matching Copilot CLI schema."""
        self.add_parameter(
            Parameter(
                name="path",
                type=ParameterType.STRING,
                description="Full absolute path to file to edit. File MUST exist.",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="old_str",
                type=ParameterType.STRING,
                description="The string in the file to replace. Preserve leading/trailing whitespace!",
                required=True,
            )
        )
        self.add_parameter(
            Parameter(
                name="new_str",
                type=ParameterType.STRING,
                description="The new string to replace old_str with.",
                required=False,
                default="",
            )
        )

    def _build_command(self, params: EditParameters) -> list[str]:
        """
        Build shell command for editing a file.

        This uses a Python script executed in the container for reliable
        string replacement with proper handling of:
        - Multi-line strings
        - Special characters
        - Exact occurrence counting
        - Atomic file updates

        Args:
            params: Typed edit parameters

        Returns:
            List of command arguments for Docker execution
        """
        # Use base64 encoding to safely pass strings with any characters
        import base64

        encoded_old = base64.b64encode(params.old_str.encode("utf-8")).decode("ascii")
        encoded_new = base64.b64encode(params.new_str.encode("utf-8")).decode("ascii")

        # Python script for safe string replacement
        # This handles all edge cases properly
        python_script = f"""
import base64
import sys

path = "{params.path}"
old_str = base64.b64decode("{encoded_old}").decode("utf-8")
new_str = base64.b64decode("{encoded_new}").decode("utf-8")

# Check if file exists
try:
    with open(path, "r") as f:
        content = f.read()
except FileNotFoundError:
    print(f"Error: File does not exist: {{path}}")
    sys.exit(1)
except Exception as e:
    print(f"Error reading file: {{e}}")
    sys.exit(1)

# Count occurrences
count = content.count(old_str)

if count == 0:
    print(f"Error: old_str not found in file")
    print(f"Searched for (length={{len(old_str)}}): {{repr(old_str[:100])}}")
    sys.exit(1)
elif count > 1:
    print(f"Error: old_str found {{count}} times in file. Must be unique for safe replacement.")
    print("Include more context in old_str to make it unique.")
    sys.exit(1)

# Perform replacement (exactly one occurrence)
new_content = content.replace(old_str, new_str, 1)

# Write back to file
try:
    with open(path, "w") as f:
        f.write(new_content)
    print(f"Successfully replaced 1 occurrence in {{path}}")
except Exception as e:
    print(f"Error writing file: {{e}}")
    sys.exit(1)
"""

        # Execute Python script in container
        return ["/bin/sh", "-c", f"python3 -c '{python_script}'"]

    async def execute(self, params: EditParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute the edit command in Docker container.

        Args:
            params: Strongly-typed edit parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with edit status
        """
        try:
            log_operation_start(
                logger,
                "edit_file",
                episode_id=context.episode_id,
                path=params.path,
                old_str_length=len(params.old_str),
                new_str_length=len(params.new_str),
            )

            timeout = int(self.get_timeout())

            # Execute the edit command
            command_args = self._build_command(params)

            try:
                result = await self._execute_in_container(context.episode_id, command_args, timeout)

                if result.exit_code == 0:
                    log_operation_success(
                        logger,
                        "edit_file",
                        episode_id=context.episode_id,
                        path=params.path,
                    )
                else:
                    log_operation_failure(
                        logger,
                        "edit_file",
                        episode_id=context.episode_id,
                        path=params.path,
                        error=result.stderr or result.stdout or "Unknown error",
                    )

                return result

            except Exception as e:
                log_operation_failure(
                    logger,
                    "edit_file",
                    episode_id=context.episode_id,
                    path=params.path,
                    error=str(e),
                )
                raise

        except SandboxExecutionError:
            raise
        except Exception as e:
            logger.error(
                "Unexpected error in edit executor",
                extra={"error": str(e)},
                exc_info=True,
            )
            return CommandResult.error_result(error=f"Edit execution error: {str(e)}")
