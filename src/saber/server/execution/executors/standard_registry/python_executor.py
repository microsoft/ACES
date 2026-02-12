"""
Docker-based Python script executor for executing Python code in isolated containers.

This module provides a secure Python executor that accepts Python code and executes
it in Docker containers with proper security validation.

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
from ...base import ExecutionContext, ExecutorParameters, Parameter, ParameterType, PythonParameters, ValidationResult
from ...exceptions import SandboxExecutionError
from ...models import ExecutorConfig, PythonExecutorConfig
from ...sandbox.sandbox_environment_manager import SandboxEnvironmentManager
from ..docker_executor import DockerExecutor

logger = get_saber_logger(LogCategory.DOCKER, __name__)


class PythonExecutor(DockerExecutor):
    """
    Docker-based Python executor for secure Python script execution.

    This executor provides Python script execution capabilities including:
    - Python code validation and syntax checking
    - Script template system for common patterns
    - Secure execution in Docker containers
    - Output parsing and error handling
    """

    _executor_metadata = {
        "name": "python",
        "description": "Execute a python command in the SABER sandbox environment.",
    }

    @classmethod
    def get_parameters_class(cls) -> type[ExecutorParameters]:
        """Get the parameter dataclass type for this executor."""
        return PythonParameters

    @classmethod
    def get_default_config(cls) -> PythonExecutorConfig:
        """
        Get default configuration for Python executor.

        Returns:
            PythonExecutorConfig with Python executor defaults
        """
        return PythonExecutorConfig(
            timeout=600.0,  # Python scripts may take longer
            allowed_modules=[
                "os",
                "sys",
                "json",
                "csv",
                "datetime",
                "time",
                "random",
                "math",
                "re",
                "collections",
                "itertools",
                "functools",
                "requests",
                "urllib",
                "base64",
                "hashlib",
                "subprocess",
            ],
            script_templates={},
        )

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxEnvironmentManager,
        config: ExecutorConfig | None = None,
        additional_params: dict[str, Any] | None = None,
        session_manager: SessionManager | None = None,
        **kwargs: Any,
    ) -> PythonExecutor:
        """
        Create Python executor with standardized configuration interface.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Typed PythonExecutorConfig
            additional_params: Additional parameters (not used for Python executor)
            **kwargs: Additional keyword arguments

        Returns:
            Configured Python executor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, **merged_kwargs)

    def __init__(
        self, sandbox_manager: SandboxEnvironmentManager, config: ExecutorConfig | None = None, **kwargs: Any
    ) -> None:
        """
        Initialize Python executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Executor configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

        # Use typed config
        typed_config = self._config if isinstance(self._config, PythonExecutorConfig) else self.get_default_config()

        # Set up allowed modules (security feature)
        self._allowed_modules = typed_config.allowed_modules
        if not self._allowed_modules:
            raise SandboxExecutionError("allowed_modules must be configured for Python executor")

        # Set up script templates
        self._script_templates = typed_config.script_templates

    def setup_parameters(self, config: ExecutorConfig) -> None:
        """Set up Python executor parameters."""
        # Python code parameter
        self.add_parameter(
            Parameter(
                name="code",
                type=ParameterType.STRING,
                description="Python code to execute in the container",
                required=True,
            )
        )

        # Template parameter for common patterns - use typed config
        typed_config = config if isinstance(config, PythonExecutorConfig) else self.get_default_config()
        script_templates = typed_config.script_templates
        self.add_parameter(
            Parameter(
                name="template",
                type=ParameterType.STRING,
                description="Optional script template to use",
                required=False,
                enum_values=list(script_templates.keys()) if script_templates else None,
            )
        )

        # Working directory parameter
        self.add_parameter(
            Parameter(
                name="working_dir",
                type=ParameterType.STRING,
                description="Working directory for script execution, must be an absolute path",
                required=False,
                default="/workspace",
            )
        )

    def validate_python_code(self, code: str) -> ValidationResult:
        """
        Validate Python code for syntax and security.

        Args:
            code: Python code string to validate

        Returns:
            ValidationResult with validation details
        """
        result = ValidationResult.success()

        if not code or not code.strip():
            result.add_error("Python code cannot be empty")
            return result

        # Check syntax
        try:
            compile(code, "<script>", "exec")
        except SyntaxError as e:
            result.add_error(f"Python syntax error: {e}")
            return result

        # Basic security checks
        dangerous_patterns = [
            "__import__",
            "exec(",
            "eval(",
            "compile(",
            "open(",
            "file(",
            "input(",
            "raw_input(",
            "globals(",
            "locals(",
            "vars(",
            "dir(",
        ]

        for pattern in dangerous_patterns:
            if pattern in code:
                result.add_warning(f"Potentially dangerous Python construct detected: {pattern}")

        # Check for restricted imports
        import ast

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self._allowed_modules:
                            result.add_warning(f"Import of non-whitelisted module: {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self._allowed_modules:
                        result.add_warning(f"Import from non-whitelisted module: {node.module}")
        except Exception as e:
            result.add_warning(f"Could not fully analyze imports: {e}")

        return result

    def _build_python_script(self, params: PythonParameters) -> str:
        """
        Build the complete Python script from parameters.

        Args:
            params: Typed Python execution parameters

        Returns:
            Complete Python script as string
        """
        code = str(params.code)  # Ensure code is a string
        template = params.template

        script: str
        if template and template in self._script_templates:
            # Use template
            template_code = self._script_templates[template]
            if "{user_code}" in template_code:
                script = template_code.replace("{user_code}", code)
            else:
                script = f"{template_code}\n\n{code}"
        else:
            # Use code directly
            script = code

        # Add any setup code
        setup_lines = []
        setup_lines.append("#!/usr/bin/env python3")
        setup_lines.append("# -*- coding: utf-8 -*-")
        setup_lines.append("import sys")
        setup_lines.append("import os")
        setup_lines.append("")

        return "\n".join(setup_lines) + script

    def get_python_environment(self, episode_id: str) -> dict[str, Any]:
        """
        Get Python environment information.

        Args:
            episode_id: Episode identifier

        Returns:
            Dictionary with Python environment details
        """
        try:
            environment = self.get_episode_environment(episode_id)
            # This would need to be implemented by running python --version etc.
            # For now, return basic info
            return {
                "python_version": "3.x",  # Would be detected from container
                "episode_id": episode_id,
                "container_ready": environment is not None,
            }
        except Exception as exc:
            logger.error(
                "Failed to collect Python environment information",
                extra={
                    "event": "python_env_inspection_failed",
                    "episode_id": episode_id,
                    "error": str(exc),
                },
            )
            return {"error": str(exc)}

    def parse_python_output(self, stdout: str, stderr: str, return_code: int, script_path: str) -> CommandResult:
        """
        Parse Python script output into a structured result.

        Args:
            stdout: Standard output from Python execution
            stderr: Standard error from Python execution
            return_code: Process exit code
            script_path: Path to the executed script

        Returns:
            CommandResult with structured output data
        """
        success = return_code == 0

        result_data = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": return_code,
            "return_code": return_code,  # Keep both for backward compatibility
            "success": success,
            "output": stdout if success else stderr,
            "script_path": script_path,
        }

        metadata = {
            "command_type": "python_script",
            "execution_environment": "docker_container",
            "exit_code": return_code,
            "has_stdout": bool(stdout.strip()),
            "has_stderr": bool(stderr.strip()),
            "output_length": len(stdout) + len(stderr),
        }

        if success:
            return CommandResult.success_result(data=result_data, metadata=metadata)
        else:
            error_msg = f"Python script failed with exit code {return_code}"
            if stderr.strip():
                error_msg += f": {stderr.strip()}"
            elif stdout.strip():
                error_msg += f". Output: {stdout.strip()}"

            return CommandResult.error_result(error=error_msg, metadata={**metadata, "raw_data": result_data})

    async def execute(self, params: PythonParameters, context: ExecutionContext) -> CommandResult:
        """
        Execute Python script in Docker container.

        Args:
            params: Strongly-typed Python parameters
            context: Strongly-typed execution context

        Returns:
            CommandResult with execution results
        """
        try:
            # Validate Python code
            code_validation = self.validate_python_code(params.code)
            if not code_validation.valid:
                return CommandResult.error_result(
                    error=f"Python code validation failed: {', '.join(code_validation.errors)}"
                )

            # Log any warnings
            if code_validation.warnings:
                for warning in code_validation.warnings:
                    logger.warning(
                        "Python code validation warning",
                        extra={
                            "event": "python_code_warning",
                            "episode_id": context.episode_id,
                            "warning": warning,
                        },
                    )

            # Build Python script
            script_content = self._build_python_script(params)

            # Write script to temporary file in container
            script_path = f"/tmp/script_{context.episode_id}.py"

            # Create script file using echo (simple approach)
            working_dir = params.working_dir
            timeout = int(self.get_timeout())
            log_operation_start(
                logger,
                "python_script_execution",
                episode_id=context.episode_id,
                timeout_seconds=timeout,
                working_dir=working_dir,
                has_template=bool(params.template),
            )
            create_script_cmd = ["sh", "-c", f"cd {working_dir} && cat > {script_path} << 'EOF'\n{script_content}\nEOF"]
            create_result = await self._execute_in_container(context.episode_id, create_script_cmd, timeout)

            if create_result.exit_code != 0:
                log_operation_failure(
                    logger,
                    "python_script_execution",
                    RuntimeError("failed_to_create_script"),
                    episode_id=context.episode_id,
                    step="create_script",
                    exit_code=create_result.exit_code,
                )
                logger.error(
                    "Failed to create Python script in container",
                    extra={
                        "event": "python_script_creation_failed",
                        "episode_id": context.episode_id,
                        "exit_code": create_result.exit_code,
                        "stderr_preview": create_result.stderr[:200],
                    },
                )
                return CommandResult.error_result(error=f"Failed to create script file: {create_result.stderr}")

            # Execute Python script
            python_cmd = ["sh", "-c", f"cd {working_dir} && python3 {script_path}"]
            result = await self._execute_in_container(context.episode_id, python_cmd, timeout)

            log_operation_success(
                logger,
                "python_script_execution",
                episode_id=context.episode_id,
                exit_code=result.exit_code,
                execution_time=result.execution_time,
            )

            # Parse output
            tool_result = self.parse_python_output(result.stdout, result.stderr, result.exit_code, script_path)

            # Add execution metadata
            container_id = self._get_container_id(context.episode_id)

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "episode_id": context.episode_id,
                    "execution_time": result.execution_time,
                }
            )

            return tool_result

        except Exception as exc:
            log_operation_failure(
                logger,
                "python_script_execution",
                exc,
                episode_id=context.episode_id,
            )
            logger.error(
                "Python script execution error",
                extra={
                    "event": "python_script_execution_error",
                    "episode_id": context.episode_id,
                    "error": str(exc),
                },
            )
            return CommandResult.error_result(f"Python execution failed: {str(exc)}")

    def validate_parameters(self, parameters: PythonParameters) -> ValidationResult:
        """
        Validate parameters for Python execution.

        Args:
            parameters: Typed PythonParameters to validate

        Returns:
            ValidationResult with comprehensive validation
        """
        # Run base Docker validation
        result = super().validate_parameters(parameters)

        # Add Python-specific validation using typed params
        code = parameters.code
        if code:
            code_validation = self.validate_python_code(code)
            result.errors.extend(code_validation.errors)
            result.warnings.extend(code_validation.warnings)

        return result


# Register this executor with the registry - must be at module level
from ..executor_registry import register_executor  # noqa: E402

register_executor("python", PythonExecutor, "standard")
