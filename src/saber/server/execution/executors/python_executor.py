"""
Docker-based Python script executor for executing Python code in isolated containers.

This module provides a secure Python executor that accepts Python code and executes
it in Docker containers with proper security validation and dependency management.
"""

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..base import CommandResult, Parameter, ParameterType, ValidationResult
from ..exceptions import SandboxExecutionError
from ..sandbox.sandbox_manager import SandboxManager
from .docker_executor import DockerExecutor

if TYPE_CHECKING:
    from ..sandbox.docker_sandbox_environment import DockerSandboxEnvironment

logger = logging.getLogger(__name__)


class PythonExecutor(DockerExecutor):
    """
    Docker-based Python executor for secure Python script execution.

    This executor provides Python script execution capabilities including:
    - Python code validation and syntax checking
    - Dependency management with pip install
    - Script template system for common patterns
    - Secure execution in Docker containers
    - Output parsing and error handling
    """

    _security_command_metadata = {
        "domain": "python",
        "name": "python_script",
        "description": "Execute Python scripts in Docker containers with dependency management",
        "author": "SABER Team",
        "security_level": "high",
        "requires_validation": True,
    }

    def __init__(
        self, sandbox_manager: SandboxManager, python_config: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> None:
        """
        Initialize Python executor.

        Args:
            sandbox_manager: Required sandbox manager for Docker execution
            python_config: Optional Python-specific configuration
            **kwargs: Additional arguments passed to parent
        """
        super().__init__(sandbox_manager=sandbox_manager, docker_config=python_config, **kwargs)

        self._python_config = python_config or {}

        # Set up allowed modules (security feature)
        self._allowed_modules = self._python_config.get(
            "allowed_modules",
            [
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
        )

        # Set up script templates
        self._script_templates = self._python_config.get("script_templates", {})

        # Add parameters
        self._setup_parameters()

    def _setup_parameters(self) -> None:
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

        # Requirements parameter for dependencies
        self.add_parameter(
            Parameter(
                name="requirements",
                type=ParameterType.ARRAY,
                description="List of Python packages to install before execution",
                required=False,
                default=[],
            )
        )

        # Template parameter for common patterns
        self.add_parameter(
            Parameter(
                name="template",
                type=ParameterType.STRING,
                description="Optional script template to use",
                required=False,
                enum_values=list(self._script_templates.keys()) if self._script_templates else None,
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

    async def install_requirements(
        self, environment: "DockerSandboxEnvironment", requirements: List[str]
    ) -> CommandResult:
        """
        Install Python requirements in the container.

        Args:
            environment: Docker environment
            requirements: List of package requirements

        Returns:
            CommandResult from pip install
        """
        if not requirements:
            return CommandResult.success_result(data={"message": "No requirements to install"})

        # Build pip install command
        pip_cmd = ["pip", "install"] + requirements

        try:
            result = environment.execute_command(command=pip_cmd)

            if result.exit_code == 0:
                return CommandResult.success_result(
                    data={
                        "stdout": result.stdout,
                        "installed_packages": requirements,
                        "execution_time": result.execution_time,
                    }
                )
            else:
                return CommandResult.error_result(
                    error=f"Package installation failed: {result.stderr}",
                    metadata={"stdout": result.stdout, "exit_code": result.exit_code},
                )
        except Exception as e:
            return CommandResult.error_result(error=f"Failed to install requirements: {e}")

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
            "return_code": return_code,
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
            parameters: Execution parameters including code and requirements
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

            # Install requirements if specified
            requirements = parameters.get("requirements", [])
            if requirements:
                install_result = await self.install_requirements(environment, requirements)
                if not install_result.success:
                    return install_result

            # Build Python script
            script_content = self.build_python_script(parameters, context)

            # Write script to temporary file in container
            script_path = f"/tmp/script_{session_id}.py"

            # Create script file using echo (simple approach)
            working_dir = parameters.get("working_dir", "/workspace")
            create_script_cmd = ["sh", "-c", f"cd {working_dir} && cat > {script_path} << 'EOF'\n{script_content}\nEOF"]
            create_result = environment.execute_command(command=create_script_cmd)

            if create_result.exit_code != 0:
                return CommandResult.error_result(error=f"Failed to create script file: {create_result.stderr}")

            # Execute Python script
            python_cmd = ["sh", "-c", f"cd {working_dir} && python3 {script_path}"]
            result = environment.execute_command(command=python_cmd)

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
                    "requirements_installed": requirements,
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
