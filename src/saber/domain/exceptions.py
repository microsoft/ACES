"""Domain-specific exceptions for SABER orchestration.

Following fail-fast philosophy - no silent failures, clear error messages.
"""


class DomainError(Exception):
    """Base exception for all domain-related errors."""

    def __init__(self, message: str, domain: str | None = None):
        super().__init__(message)
        self.domain = domain
        self.message = message


class DomainNotFoundError(DomainError):
    """Raised when a domain cannot be found."""

    def __init__(self, domain: str, domains_root: str):
        message = (
            f"Domain '{domain}' not found in {domains_root}. "
            f"Available domains can be listed with 'python -m saber.domain list'."
        )
        super().__init__(message, domain)
        self.domains_root = domains_root


class DomainValidationError(DomainError):
    """Raised when domain configuration is invalid."""

    def __init__(self, domain: str, validation_errors: list[str]):
        message = f"Domain '{domain}' configuration is invalid:\n" + "\n".join(
            f"  - {error}" for error in validation_errors
        )
        super().__init__(message, domain)
        self.validation_errors = validation_errors


class ResourceNotFoundError(DomainError):
    """Raised when required resources cannot be found."""

    def __init__(self, resource_type: str, details: str):
        message = (
            f"Required {resource_type} not found: {details}. "
            f"Ensure SABER is properly installed or provide explicit paths."
        )
        super().__init__(message)
        self.resource_type = resource_type


class DockerError(DomainError):
    """Raised when Docker operations fail."""

    def __init__(self, message: str, command: str | None = None, exit_code: int | None = None):
        if command:
            full_message = f"Docker command failed: {command}\n{message}"
        else:
            full_message = f"Docker operation failed: {message}"

        super().__init__(full_message)
        self.command = command
        self.exit_code = exit_code
