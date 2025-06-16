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
MAX_BYTES = 10 * 1024 * 1024  # 10MB
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

    def filter(self, record: logging.LogRecord) -> bool:
        """Add request_id to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        record.request_id = getattr(record, "request_id", "N/A")
        return True


class ContextFilter(logging.Filter):
    """Filter to add context information to log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Add context information to the log record.

        Args:
            record: The log record to modify.

        Returns:
            True to allow the record to be processed.
        """
        if not hasattr(record, "context"):
            record.context = {}
        return True


class JSONFormatter(logging.Formatter):
    """Formatter that outputs JSON strings after parsing the LogRecord."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            JSON-formatted string representation of the log record.
        """
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "request_id": getattr(record, "request_id", "N/A"),
            "context": getattr(record, "context", {}),
        }

        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info),
            }

        return json.dumps(log_data)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: Name of the logger.

    Returns:
        Logger instance with context filter.
    """
    logger = logging.getLogger(name)
    logger.addFilter(ContextFilter())
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
    extra = {"context": context or {}}
    logger.log(level, message, extra=extra, exc_info=exc_info)


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = DEFAULT_LOG_LEVEL,
    log_format: str = DEFAULT_LOG_FORMAT,
    app_name: str = "arklex",
    use_json: bool = False,
) -> None:
    """Set up logging configuration for the application.

    Args:
        log_dir: Directory to store log files. If None, logs will be stored in 'logs' directory.
        log_level: Logging level (default: INFO).
        log_format: Format string for log messages.
        app_name: Name of the application for log file naming.
        use_json: Whether to use JSON formatting for logs.
    """
    # Create logs directory if it doesn't exist
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if use_json:
        file_formatter = JSONFormatter()
        console_formatter = JSONFormatter()
    else:
        file_formatter = logging.Formatter(log_format)
        console_formatter = logging.Formatter("%(levelname)s - %(message)s")

    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=MAX_BYTES, backupCount=BACKUP_COUNT, encoding="utf-8"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.addFilter(RequestIdFilter())
    file_handler.addFilter(ContextFilter())
    root_logger.addHandler(file_handler)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(console_formatter)
    console_handler.addFilter(RequestIdFilter())
    console_handler.addFilter(ContextFilter())
    root_logger.addHandler(console_handler)

    # Set logging levels for specific modules
    for module, level in MODULE_LOG_LEVELS.items():
        logging.getLogger(module).setLevel(level)
