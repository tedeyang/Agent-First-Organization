"""Exception definitions for the Arklex framework.

This module defines custom exceptions used throughout the Arklex framework,
including authentication errors, validation errors, and user-facing errors.
It provides a base class for all exceptions and implements specific error types
for different scenarios.
"""

from typing import Optional, Any, Dict


class ArklexError(Exception):
    """Base exception class for all Arklex errors.

    Attributes:
        message: A human-readable error message
        code: An error code for programmatic handling
        status_code: HTTP status code for API responses
        details: Additional error details
    """

    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            code: An error code for programmatic handling
            status_code: HTTP status code for API responses
            details: Additional error details
        """
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        """Get a string representation of the error.

        Returns:
            str: The error message with code if present
        """
        if self.code:
            return f"{self.message} (code: {self.code})"
        return self.message


class AuthenticationError(ArklexError):
    """Exception raised for authentication errors."""

    def __init__(
        self,
        message: str = "Authentication failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="AUTHENTICATION_ERROR",
            status_code=401,
            details=details,
        )


class ValidationError(ArklexError):
    """Exception raised for validation errors."""

    def __init__(
        self,
        message: str = "Validation failed",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the exception.

        Args:
            message: A human-readable error message
            details: Additional error details
        """
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=details,
        )


class APIError(ArklexError):
    """Raised when API calls fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the APIError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "API_ERROR", 500, details)


class ModelError(ArklexError):
    """Raised when model operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ModelError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "MODEL_ERROR", 500, details)


class ConfigurationError(ArklexError):
    """Raised when there are configuration issues.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ConfigurationError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "CONFIG_ERROR", 500, details)


class DatabaseError(ArklexError):
    """Raised when database operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the DatabaseError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "DB_ERROR", 500, details)


class ResourceNotFoundError(ArklexError):
    """Raised when a requested resource is not found.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ResourceNotFoundError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "NOT_FOUND", 404, details)


class RateLimitError(ArklexError):
    """Raised when rate limits are exceeded.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the RateLimitError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "RATE_LIMIT", 429, details)


class ToolExecutionError(ArklexError):
    """Raised when tool execution fails.

    Args:
        tool_name: Name of the tool that failed.
        message: The error message.
        details: Optional dictionary with additional error details.
        extra_message: Optional additional message for user guidance.
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        extra_message: Optional[str] = None,
    ) -> None:
        """Initialize the ToolExecutionError.

        Args:
            tool_name: Name of the tool that failed.
            message: The error message.
            details: Optional dictionary with additional error details.
            extra_message: Optional additional message for user guidance.
        """
        super().__init__(
            f"Tool {tool_name} execution failed: {message}",
            "TOOL_ERROR",
            500,
            details,
        )


class UserFacingError(ArklexError):
    """Base class for user-facing errors with additional guidance.

    Args:
        message: The error message.
        error_code: Error code for categorization.
        details: Optional dictionary with additional error details.
        extra_message: Optional additional message for user guidance.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        extra_message: Optional[str] = None,
    ) -> None:
        """Initialize the UserFacingError.

        Args:
            message: The error message.
            error_code: Error code for categorization.
            details: Optional dictionary with additional error details.
            extra_message: Optional additional message for user guidance.
        """
        super().__init__(message, error_code, 500, details)


class RetryableError(ArklexError):
    """Base class for errors that can be retried.

    Args:
        message: The error message.
        error_code: Error code for categorization.
        details: Optional dictionary with additional error details.
        max_retries: Maximum number of retry attempts.
    """

    def __init__(
        self,
        message: str,
        error_code: str,
        details: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
    ) -> None:
        """Initialize the RetryableError.

        Args:
            message: The error message.
            error_code: Error code for categorization.
            details: Optional dictionary with additional error details.
            max_retries: Maximum number of retry attempts.
        """
        super().__init__(message, error_code, 500, details)
        self.max_retries = max_retries


class NetworkError(RetryableError):
    """Raised when network operations fail.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the NetworkError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "NETWORK_ERROR", details)


class TimeoutError(RetryableError):
    """Raised when operations timeout.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the TimeoutError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "TIMEOUT_ERROR", details)


class ServiceUnavailableError(RetryableError):
    """Raised when a service is temporarily unavailable.

    Args:
        message: The error message.
        details: Optional dictionary with additional error details.
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the ServiceUnavailableError.

        Args:
            message: The error message.
            details: Optional dictionary with additional error details.
        """
        super().__init__(message, "SERVICE_UNAVAILABLE", details)
