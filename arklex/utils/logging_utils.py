"""Logging utilities for the Arklex framework.

This module provides standardized logging patterns, context management,
and error handling utilities to ensure consistent logging across the application.
"""

import asyncio
import functools
import logging
import inspect
import traceback
from typing import Any, Callable, Dict, Optional, Type, Union, List
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from arklex.utils.logging_config import get_logger, log_with_context
from arklex.utils.exceptions import ArklexError, RetryableError

# Standard log messages for consistent usage
LOG_MESSAGES = {
    "ERROR": {
        "API_FAILURE": "API call failed: {error}",
        "VALIDATION_ERROR": "Validation failed: {error}",
        "UNEXPECTED_ERROR": "Unexpected error in {function}: {error}",
        "OPERATION_FAILED": "Operation failed: {error}",
        "RETRY_FAILED": "Retry failed after {attempts} attempts: {error}",
        "RESOURCE_ERROR": "Resource error: {error}",
        "PERMISSION_ERROR": "Permission denied: {error}",
        "CONFIGURATION_ERROR": "Configuration error: {error}",
    },
    "WARNING": {
        "RETRY_ATTEMPT": "Retrying operation (attempt {attempt}/{max_attempts}): {error}",
        "DEPRECATED_FEATURE": "Deprecated feature used: {feature}",
        "PERFORMANCE_WARNING": "Performance warning: {message}",
        "RESOURCE_WARNING": "Resource warning: {message}",
        "CONFIGURATION_WARNING": "Configuration warning: {message}",
    },
    "INFO": {
        "REQUEST_START": "Request started: {request_id}",
        "REQUEST_END": "Request completed: {request_id}",
        "OPERATION_START": "Starting operation: {operation}",
        "OPERATION_END": "Completed operation: {operation}",
        "RESOURCE_ACCESS": "Accessing resource: {resource}",
        "CONFIGURATION_LOAD": "Loading configuration: {config}",
    },
    "DEBUG": {
        "FUNCTION_ENTRY": "Entering function: {function}",
        "FUNCTION_EXIT": "Exiting function: {function}",
        "STATE_CHANGE": "State changed: {old_state} -> {new_state}",
        "RESOURCE_DETAILS": "Resource details: {details}",
        "CONFIGURATION_DETAILS": "Configuration details: {details}",
    },
}

logger = get_logger(__name__)


class LogContext:
    """Context manager for structured logging with consistent context."""

    def __init__(
        self, logger_name: str, context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Initialize the log context.

        Args:
            logger_name: Name of the logger to use
            context: Base context to include in all logs
        """
        self.logger = get_logger(logger_name)
        self.context = context or {}
        self._stack: List[Dict[str, Any]] = []

    def push_context(self, additional_context: Dict[str, Any]) -> None:
        """Push additional context onto the stack.

        Args:
            additional_context: Additional context to add
        """
        self._stack.append(additional_context)
        self.context = {**self.context, **additional_context}

    def pop_context(self) -> None:
        """Pop the most recent context from the stack."""
        if self._stack:
            self._stack.pop()
            # Rebuild context from remaining stack
            self.context = {}
            for ctx in self._stack:
                self.context.update(ctx)

    def log(self, level: Union[str, int], message: str, **kwargs: Any) -> None:
        """Log a message with context.

        Args:
            level: Log level (string or integer)
            message: Log message
            **kwargs: Additional context to include
        """
        exc_info = kwargs.pop("exc_info", None)
        log_with_context(
            self.logger,
            level,
            message,
            context={**self.context, **kwargs},
            exc_info=exc_info,
        )

    def error(
        self, message: str, exc_info: Optional[Exception] = None, **kwargs: Any
    ) -> None:
        """Log an error with context.

        Args:
            message: Error message
            exc_info: Exception information
            **kwargs: Additional context
        """
        error_context = {
            "error_type": type(exc_info).__name__ if exc_info else None,
            "error_details": getattr(exc_info, "details", None) if exc_info else None,
            "stack_trace": traceback.format_exc() if exc_info else None,
        }
        self.log("ERROR", message, exc_info=exc_info, **{**error_context, **kwargs})

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning with context.

        Args:
            message: Warning message
            **kwargs: Additional context
        """
        self.log("WARNING", message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message with context.

        Args:
            message: Info message
            **kwargs: Additional context
        """
        self.log("INFO", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message with context.

        Args:
            message: Debug message
            **kwargs: Additional context
        """
        self.log("DEBUG", message, **kwargs)


def handle_exceptions(
    *,
    reraise: bool = True,
    default_error: Type[ArklexError] = ArklexError,
    log_level: str = "ERROR",
    include_stack_trace: bool = True,
) -> Callable:
    """Decorator for consistent exception handling and logging.

    Args:
        reraise: Whether to re-raise the exception
        default_error: Default error type to use for unknown exceptions
        log_level: Log level to use for error logging
        include_stack_trace: Whether to include stack trace in logs

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except ArklexError:
                raise
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                if include_stack_trace:
                    context["stack_trace"] = traceback.format_exc()

                log_with_context(
                    get_logger(func.__module__),
                    log_level,
                    LOG_MESSAGES["ERROR"]["UNEXPECTED_ERROR"].format(
                        function=func.__name__, error=str(e)
                    ),
                    context=context,
                    exc_info=e,
                )
                if reraise:
                    raise default_error(
                        f"Operation failed in {func.__name__}",
                        details={
                            "original_error": str(e),
                            "error_type": type(e).__name__,
                            "module": func.__module__,
                        },
                    ) from e
                return None

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ArklexError:
                raise
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "module": func.__module__,
                    "args": str(args),
                    "kwargs": str(kwargs),
                }
                if include_stack_trace:
                    context["stack_trace"] = traceback.format_exc()

                log_with_context(
                    get_logger(func.__module__),
                    log_level,
                    LOG_MESSAGES["ERROR"]["UNEXPECTED_ERROR"].format(
                        function=func.__name__, error=str(e)
                    ),
                    context=context,
                    exc_info=e,
                )
                if reraise:
                    raise default_error(
                        f"Operation failed in {func.__name__}",
                        details={
                            "original_error": str(e),
                            "error_type": type(e).__name__,
                            "module": func.__module__,
                        },
                    ) from e
                return None

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator


def with_retry(
    *,
    max_attempts: int = 3,
    min_wait: int = 1,
    max_wait: int = 10,
    retry_on: Optional[Type[Exception]] = None,
    include_stack_trace: bool = True,
) -> Callable:
    """Decorator for adding retry logic to functions.

    Args:
        max_attempts: Maximum number of retry attempts
        min_wait: Minimum wait time between retries (seconds)
        max_wait: Maximum wait time between retries (seconds)
        retry_on: Exception type(s) to retry on (defaults to RetryableError)
        include_stack_trace: Whether to include stack trace in logs

    Returns:
        Decorated function
    """

    def decorator(func: Callable) -> Callable:
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on or RetryableError),
        )
        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, (retry_on or RetryableError)):
                    context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "attempt": getattr(e, "attempt", 0),
                        "max_attempts": max_attempts,
                    }
                    if include_stack_trace:
                        context["stack_trace"] = traceback.format_exc()

                    log_with_context(
                        get_logger(func.__module__),
                        "WARNING",
                        LOG_MESSAGES["WARNING"]["RETRY_ATTEMPT"].format(
                            attempt=getattr(e, "attempt", 0),
                            max_attempts=max_attempts,
                            error=str(e),
                        ),
                        context=context,
                    )
                raise

        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
            retry=retry_if_exception_type(retry_on or RetryableError),
        )
        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if isinstance(e, (retry_on or RetryableError)):
                    context = {
                        "function": func.__name__,
                        "module": func.__module__,
                        "args": str(args),
                        "kwargs": str(kwargs),
                        "attempt": getattr(e, "attempt", 0),
                        "max_attempts": max_attempts,
                    }
                    if include_stack_trace:
                        context["stack_trace"] = traceback.format_exc()

                    log_with_context(
                        get_logger(func.__module__),
                        "WARNING",
                        LOG_MESSAGES["WARNING"]["RETRY_ATTEMPT"].format(
                            attempt=getattr(e, "attempt", 0),
                            max_attempts=max_attempts,
                            error=str(e),
                        ),
                        context=context,
                    )
                raise

        return async_wrapper if inspect.iscoroutinefunction(func) else sync_wrapper

    return decorator
