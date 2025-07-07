"""
Common fixtures and configuration for integration tests.

This module provides shared fixtures and configuration for all integration tests,
including proper environment setup and common test utilities.
"""

import os
import sys
from pathlib import Path
from typing import Any

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set up common environment variables for integration tests
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("MYSQL_USERNAME", "test_user")
os.environ.setdefault("MYSQL_PASSWORD", "test_password")
os.environ.setdefault("MYSQL_HOSTNAME", "localhost")
os.environ.setdefault("MYSQL_PORT", "3306")
os.environ.setdefault("MYSQL_DB_NAME", "test_db")


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Provide the test data directory path."""
    return project_root / "examples" / "hitl_server"


@pytest.fixture(scope="session")
def hitl_taskgraph_path(test_data_dir: Path) -> Path:
    """Provide the path to the HITL taskgraph configuration."""
    return test_data_dir / "taskgraph.json"


@pytest.fixture(scope="session")
def load_hitl_config(hitl_taskgraph_path: Path) -> dict[str, Any]:
    """Load the HITL taskgraph configuration."""
    import json

    with open(hitl_taskgraph_path, encoding="utf-8") as f:
        config = json.load(f)

    # Set up model configuration for testing
    model = {
        "model_name": "gpt-3.5-turbo",
        "model_type_or_path": "gpt-3.5-turbo",
        "llm_provider": "openai",
        "api_key": "test_key",
        "endpoint": "https://api.openai.com/v1",
    }
    config["model"] = model
    config["input_dir"] = str(hitl_taskgraph_path.parent)

    return config


@pytest.fixture(scope="session")
def mock_environment_variables() -> None:
    """Set up mock environment variables for testing."""
    original_env = os.environ.copy()

    # Set test environment variables
    test_env_vars = {
        "OPENAI_API_KEY": "test_key",
        "DATA_DIR": "./examples/hitl_server",
        "MYSQL_USERNAME": "test_user",
        "MYSQL_PASSWORD": "test_password",
        "MYSQL_HOSTNAME": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_DB_NAME": "test_db",
    }

    for key, value in test_env_vars.items():
        os.environ[key] = value

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(scope="function")
def clean_test_state() -> None:
    """Ensure clean state for each test function."""
    # This fixture can be used to reset any state between tests
    yield
    # Cleanup can be added here if needed


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "hitl: marks tests as HITL (Human-in-the-Loop) tests"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Modify test collection to add markers based on test location."""
    for item in items:
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Mark HITL tests
        if "hitl" in str(item.fspath) or "hitl" in item.name:
            item.add_marker(pytest.mark.hitl)

        # Mark slow tests (can be customized based on test names)
        if any(
            slow_indicator in item.name.lower()
            for slow_indicator in ["slow", "integration", "e2e"]
        ):
            item.add_marker(pytest.mark.slow)
