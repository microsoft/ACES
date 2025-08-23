"""
Docker-based Python script executor for executing Python code in isolated containers.

This module provides a secure Python executor that accepts Python code and executes
it in Docker containers with proper security validation.
"""

import logging
from typing import Any, Dict, Optional

from ....base import CommandResult
from ...base import Parameter, ParameterType, ValidationResult
from ...exceptions import SandboxExecutionError
from ...sandbox.sandbox_manager import SandboxManager
from ..docker_executor import DockerExecutor

logger = logging.getLogger(__name__)


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
        "name": "execute_python",
        "description": "Execute Python code in secure Docker container",
    }

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration for Python executor.

        Returns:
            Dictionary containing Python executor default configuration
        """
        return {
            "allowed_modules": [
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
            "script_templates": {},
            "timeout": 600.0,  # Python scripts may take longer
        }

    @classmethod
    def create_with_config(
        cls,
        sandbox_manager: SandboxManager,
        config: Optional[Dict[str, Any]] = None,
        additional_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> "PythonExecutor":
        """
        Create Python executor with standardized configuration interface.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Python-specific configuration dictionary
            additional_params: Additional parameters (not used for Python executor)
            **kwargs: Additional keyword arguments

        Returns:
            Configured Python executor instance
        """
        merged_kwargs = {**kwargs}
        if additional_params:
            merged_kwargs.update(additional_params)

        return cls(sandbox_manager=sandbox_manager, config=config, **merged_kwargs)

    def __init__(self, sandbox_manager: SandboxManager, config: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        """
        Initialize Python executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            config: Python-specific configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, config=config, **kwargs)

        # Set up allowed modules (security feature) - must be explicitly configured
        if "allowed_modules" not in self._config:
            raise SandboxExecutionError("allowed_modules must be explicitly configured for Python executor")

        self._allowed_modules = self._config["allowed_modules"]
        if not isinstance(self._allowed_modules, list):
            raise SandboxExecutionError("allowed_modules must be a list of module names")

        # Set up script templates
        self._script_templates = self._config.get("script_templates", {})

    def setup_parameters(self, config: Dict[str, Any]) -> None:
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

        # Template parameter for common patterns
        script_templates = config.get("script_templates", {})
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
                description="Working directory for script execution",
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

    def build_python_script(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> str:
        """
        Build the complete Python script from parameters.

        Args:
            parameters: Execution parameters
            context: Execution context

        Returns:
            Complete Python script as string
        """
        code = str(parameters["code"])  # Ensure code is a string
        template = parameters.get("template")

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

    def get_python_environment(self, session_id: str) -> Dict[str, Any]:
        """
        Get Python environment information.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with Python environment details
        """
        try:
            environment = self.get_session_environment(session_id)
            # This would need to be implemented by running python --version etc.
            # For now, return basic info
            return {
                "python_version": "3.x",  # Would be detected from container
                "session_id": session_id,
                "container_ready": environment is not None,
            }
        except Exception as e:
            logger.error(f"Failed to get Python environment info: {e}")
            return {"error": str(e)}

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

    async def execute(self, parameters: Dict[str, Any], context: Dict[str, Any]) -> CommandResult:
        """
        Execute Python script in Docker container.

        Args:
            parameters: Execution parameters including code
            context: Execution context including session_id

        Returns:
            CommandResult with execution results
        """
        try:
            # Extract session ID
            session_id = context.get("session_id")
            if not session_id:
                raise SandboxExecutionError("session_id required in context for Python execution")

            # Get Docker environment
            environment = self.get_session_environment(session_id)

            # Validate Python code
            code_validation = self.validate_python_code(parameters["code"])
            if not code_validation.valid:
                return CommandResult.error_result(
                    error=f"Python code validation failed: {', '.join(code_validation.errors)}"
                )

            # Log any warnings
            if code_validation.warnings:
                logger.warning(f"Python code warnings: {', '.join(code_validation.warnings)}")

            # Build Python script
            script_content = self.build_python_script(parameters, context)

            # Write script to temporary file in container
            script_path = f"/tmp/script_{session_id}.py"

            # Create script file using echo (simple approach)
            working_dir = parameters.get("working_dir", "/workspace")
            timeout = int(self.get_timeout())
            create_script_cmd = ["sh", "-c", f"cd {working_dir} && cat > {script_path} << 'EOF'\n{script_content}\nEOF"]
            create_result = environment.execute_command(command=create_script_cmd, timeout=timeout)

            if create_result.exit_code != 0:
                return CommandResult.error_result(error=f"Failed to create script file: {create_result.stderr}")

            # Execute Python script
            python_cmd = ["sh", "-c", f"cd {working_dir} && python3 {script_path}"]
            result = environment.execute_command(command=python_cmd, timeout=timeout)

            # Parse output
            tool_result = self.parse_python_output(result.stdout, result.stderr, result.exit_code, script_path)

            # Add execution metadata
            container = environment.get_execution_container()
            container_id = container.id[:12] if container else "unknown"

            tool_result.metadata.update(
                {
                    "container_id": container_id,
                    "session_id": session_id,
                    "execution_time": result.execution_time,
                }
            )

            return tool_result

        except Exception as e:
            logger.error(f"Python script execution error: {e}")
            return CommandResult.error_result(f"Python execution failed: {str(e)}")

    def validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """
        Validate parameters for Python execution.

        Args:
            parameters: Parameters to validate

        Returns:
            ValidationResult with comprehensive validation
        """
        # Run base Docker validation
        result = super().validate_parameters(parameters)

        # Add Python-specific validation
        if "code" in parameters:
            code_validation = self.validate_python_code(parameters["code"])
            result.errors.extend(code_validation.errors)
            result.warnings.extend(code_validation.warnings)

        # Validate requirements
        requirements = parameters.get("requirements", [])
        if requirements and not isinstance(requirements, list):
            result.add_error("requirements must be a list of strings")
        elif requirements:
            for req in requirements:
                if not isinstance(req, str):
                    result.add_error(f"requirement must be string, got: {type(req)}")

        return result


# Register this executor with the registry - must be at module level
from ..executor_registry import register_executor  # noqa: E402

register_executor("python", PythonExecutor, "standard")
