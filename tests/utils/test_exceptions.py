import pytest
from arklex.utils.exceptions import (
    ArklexError,
    AuthenticationError,
    ValidationError,
    APIError,
    ModelError,
    ConfigurationError,
    DatabaseError,
    ResourceNotFoundError,
    RateLimitError,
)


def test_arklex_error_base():
    """Test the base ArklexError class."""
    error = ArklexError("Test error")
    assert str(error) == "Test error"
    assert error.error_code is None
    assert error.details == {}


def test_arklex_error_with_code():
    """Test ArklexError with error code."""
    error = ArklexError("Test error", error_code="TEST_ERROR")
    assert str(error) == "[TEST_ERROR] Test error"
    assert error.error_code == "TEST_ERROR"


def test_arklex_error_with_details():
    """Test ArklexError with details."""
    details = {"test": "detail"}
    error = ArklexError("Test error", details=details)
    assert error.details == details


def test_authentication_error():
    """Test AuthenticationError."""
    error = AuthenticationError("Auth failed")
    assert str(error) == "[AUTH_ERROR] Auth failed"
    assert error.error_code == "AUTH_ERROR"


def test_validation_error():
    """Test ValidationError."""
    error = ValidationError("Invalid input")
    assert str(error) == "[VALIDATION_ERROR] Invalid input"
    assert error.error_code == "VALIDATION_ERROR"


def test_api_error():
    """Test APIError."""
    error = APIError("API call failed")
    assert str(error) == "[API_ERROR] API call failed"
    assert error.error_code == "API_ERROR"


def test_model_error():
    """Test ModelError."""
    error = ModelError("Model failed")
    assert str(error) == "[MODEL_ERROR] Model failed"
    assert error.error_code == "MODEL_ERROR"


def test_configuration_error():
    """Test ConfigurationError."""
    error = ConfigurationError("Config invalid")
    assert str(error) == "[CONFIG_ERROR] Config invalid"
    assert error.error_code == "CONFIG_ERROR"


def test_database_error():
    """Test DatabaseError."""
    error = DatabaseError("DB operation failed")
    assert str(error) == "[DB_ERROR] DB operation failed"
    assert error.error_code == "DB_ERROR"


def test_resource_not_found_error():
    """Test ResourceNotFoundError."""
    error = ResourceNotFoundError("Resource not found")
    assert str(error) == "[NOT_FOUND] Resource not found"
    assert error.error_code == "NOT_FOUND"


def test_rate_limit_error():
    """Test RateLimitError."""
    error = RateLimitError("Rate limit exceeded")
    assert str(error) == "[RATE_LIMIT] Rate limit exceeded"
    assert error.error_code == "RATE_LIMIT"


def test_error_inheritance():
    """Test that all custom errors inherit from ArklexError."""
    error_classes = [
        AuthenticationError,
        ValidationError,
        APIError,
        ModelError,
        ConfigurationError,
        DatabaseError,
        ResourceNotFoundError,
        RateLimitError,
    ]

    for error_class in error_classes:
        error = error_class("Test")
        assert isinstance(error, ArklexError)
