"""Logging configuration for the Arklex framework.

This module provides centralized logging configuration, including:
- Log level management
- Log formatting
- Log rotation
- Request ID tracking
- Context-aware logging
- JSON logging support
"""

import logging
import logging.handlers
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union

# Constants
DEFAULT_LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
)
DEFAULT_LOG_LEVEL = logging.INFO
MAX_BYTES = 1024 * 1024  # 1MB for production, but tests will override this
BACKUP_COUNT = 5

# Standard log levels for different types of events
LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,  # System is unusable
    "ERROR": logging.ERROR,  # Error events that might still allow the application to continue
    "WARNING": logging.WARNING,  # Warning events that might indicate a problem
    "INFO": logging.INFO,  # General information about program execution
    "DEBUG": logging.DEBUG,  # Detailed information for debugging
}

# Module-specific log levels
MODULE_LOG_LEVELS = {
    "urllib3": logging.WARNING,
    "requests": logging.WARNING,
    "fastapi": logging.INFO,
    "uvicorn": logging.INFO,
    "sqlalchemy": logging.WARNING,
    "httpx": logging.WARNING,
    "websockets": logging.WARNING,
}


class RequestIdFilter(logging.Filter):
    """Filter to add request_id to log records."""

    def __init__(self, request_id: str = "N/A") -> None:
        """Initialize the filter.

        Args:
            request_id: The request ID to add to log records
        """
        super().__init__()
        self.request_id = request_id

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        record.request_id = self.request_id
        return True


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""

    def __init__(self, context: Optional[Dict[str, Any]] = None) -> None:
        """Initialize the filter.

        Args:
            context: Dictionary of context information to add to log records
        """
        super().__init__()
        self.context = context or {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        record.context = self.context
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the LogRecord."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as JSON.

        Args:
            record: The log record to format.

        Returns:
            JSON string representation of the log record.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id

        if hasattr(record, "context"):
            log_data["context"] = record.context

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def get_logger(
    name: str,
    level: Optional[Union[str, int]] = None,
    log_format: Optional[str] = None,
) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Name of the logger.
        level: Optional log level (string or integer).
        log_format: Optional format string for log messages.

    Returns:
        Logger instance with context filter.
    """
    logger = logging.getLogger(name)

    # Set log level if provided
    if level is not None:
        if isinstance(level, str):
            level = LOG_LEVELS.get(level.upper(), logging.INFO)
        logger.setLevel(level)

    # Add handlers if none exist
    if not logger.handlers:
        # Inherit formatter from parent if exists and no log_format is provided
        parent = logger.parent if logger.parent and logger.parent != logger else None
        if log_format is not None:
            formatter = logging.Formatter(log_format)
        elif parent and parent.handlers:
            formatter = parent.handlers[0].formatter
        else:
            formatter = logging.Formatter(DEFAULT_LOG_FORMAT)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(RequestIdFilter())
        console_handler.addFilter(ContextFilter())
        logger.addHandler(console_handler)

        # Set propagation to False to prevent duplicate logs
        logger.propagate = False
    else:
        # If log_format is provided, update formatter
        if log_format is not None:
            for handler in logger.handlers:
                handler.setFormatter(logging.Formatter(log_format))
        # If no log_format but parent has a formatter, copy it
        elif logger.parent and logger.parent != logger and logger.parent.handlers:
            parent_formatter = logger.parent.handlers[0].formatter
            for handler in logger.handlers:
                handler.setFormatter(parent_formatter)

    return logger


def log_with_context(
    logger: logging.Logger,
    level: Union[str, int],
    message: str,
    context: Optional[Dict[str, Any]] = None,
    exc_info: Optional[Exception] = None,
) -> None:
    """Log a message with context information.

    Args:
        logger: Logger instance.
        level: Log level (string or integer).
        message: Log message.
        context: Additional context information.
        exc_info: Exception information.
    """
    if isinstance(level, str):
        level = LOG_LEVELS.get(level.upper(), logging.INFO)

    # Temporarily add ContextFilter to all handlers
    filters = []
    for handler in logger.handlers:
        context_filter = ContextFilter(context or {})
        handler.addFilter(context_filter)
        filters.append((handler, context_filter))
    try:
        logger.log(level, message, exc_info=exc_info)
    finally:
        for handler, context_filter in filters:
            handler.removeFilter(context_filter)


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: Union[str, int] = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    app_name: str = "arklex",
    use_json: bool = False,
    max_bytes: int = MAX_BYTES,
) -> None:
    """Set up logging configuration for the application.

    Args:
        log_dir: Directory to store log files. If None, logs will be stored in 'logs' directory.
        log_level: Logging level (default: INFO).
        log_format: Format string for log messages.
        app_name: Name of the application for log file naming.
        use_json: Whether to use JSON formatting for logs.
        max_bytes: Maximum bytes for log rotation.
    """
    root_logger = logging.getLogger()
    # Remove all existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # Convert string log level to integer if needed
    if isinstance(log_level, str):
        log_level = LOG_LEVELS.get(log_level.upper(), DEFAULT_LOG_LEVEL)

    # Set root logger level
    root_logger.setLevel(log_level)

    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Create formatter
    if use_json:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(log_format)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.addFilter(RequestIdFilter())
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)

    # Create file handler with rotation
    log_file = os.path.join(log_dir, f"{app_name}.log")
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=max_bytes,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    file_handler.addFilter(RequestIdFilter())
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    # Configure module-specific log levels
    for module, level in MODULE_LOG_LEVELS.items():
        logging.getLogger(module).setLevel(level)

    # Disable propagation for root logger to prevent duplicate logs
    root_logger.propagate = False
