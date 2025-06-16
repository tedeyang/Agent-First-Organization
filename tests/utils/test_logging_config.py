import os
import logging
import pytest
from arklex.utils.logging_config import setup_logging, get_logger, RequestIdFilter


def test_setup_logging_creates_log_directory(tmp_path):
    """Test that setup_logging creates the log directory."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    assert os.path.exists(log_dir)


def test_setup_logging_creates_log_file(tmp_path):
    """Test that setup_logging creates a log file."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    log_files = os.listdir(log_dir)
    assert len(log_files) > 0
    assert any(f.endswith(".log") for f in log_files)


def test_get_logger_returns_logger():
    """Test that get_logger returns a logger instance."""
    logger = get_logger("test_logger")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_logger"


def test_request_id_filter():
    """Test the RequestIdFilter adds request_id to log records."""
    filter_instance = RequestIdFilter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname="test.py",
        lineno=1,
        msg="Test message",
        args=(),
        exc_info=None,
    )

    # Test with no request_id
    assert filter_instance.filter(record)
    assert record.request_id == "N/A"

    # Test with existing request_id
    record.request_id = "test-123"
    assert filter_instance.filter(record)
    assert record.request_id == "test-123"


def test_logging_levels(tmp_path):
    """Test that different logging levels work correctly."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir, log_level=logging.DEBUG)
    logger = get_logger("test_logger")

    # Test all logging levels
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

    # Read the log file
    log_file = os.path.join(log_dir, os.listdir(log_dir)[0])
    with open(log_file, "r") as f:
        log_content = f.read()

    # Verify all levels are present
    assert "DEBUG" in log_content
    assert "INFO" in log_content
    assert "WARNING" in log_content
    assert "ERROR" in log_content


def test_log_rotation(tmp_path):
    """Test that log rotation works correctly."""
    log_dir = str(tmp_path / "test_logs")
    setup_logging(log_dir=log_dir)
    logger = get_logger("test_logger")

    # Generate enough log entries to trigger rotation
    for i in range(1000):
        logger.info("x" * 1000)  # Large log entries

    # Check that rotation files were created
    log_files = os.listdir(log_dir)
    assert len(log_files) > 1  # Should have at least one rotated file
