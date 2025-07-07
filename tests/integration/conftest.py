"""
Common fixtures and configuration for integration tests.

This module provides shared fixtures and configuration for all integration tests,
including proper environment setup and common test utilities.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Mock the mysql module BEFORE any other imports to prevent connection issues
sys.modules["arklex.utils.mysql"] = MagicMock()

# Set up common environment variables for integration tests
os.environ.setdefault("OPENAI_API_KEY", "test_key")
os.environ.setdefault("MYSQL_USERNAME", "test_user")
os.environ.setdefault("MYSQL_PASSWORD", "test_password")
os.environ.setdefault("MYSQL_HOSTNAME", "localhost")
os.environ.setdefault("MYSQL_PORT", "3306")
os.environ.setdefault("MYSQL_DB_NAME", "test_db")
os.environ.setdefault("DATA_DIR", "./examples/hitl_server")
os.environ.setdefault("ARKLEX_TEST_ENV", "local")
os.environ.setdefault("TESTING", "true")
os.environ.setdefault("LOG_LEVEL", "WARNING")


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
        "ARKLEX_TEST_ENV": "local",
        "TESTING": "true",
        "LOG_LEVEL": "WARNING",
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


# Shopify-specific fixtures
@pytest.fixture
def mock_shopify_session() -> Mock:
    """Create a mock Shopify session for testing."""
    mock_session = Mock()
    mock_session.__enter__ = Mock(return_value=mock_session)
    mock_session.__exit__ = Mock(return_value=None)
    return mock_session


@pytest.fixture
def mock_shopify_graphql() -> Mock:
    """Create a mock Shopify GraphQL client for testing."""
    mock_graphql = Mock()
    mock_graphql_instance = Mock()
    mock_graphql_instance.execute.return_value = json.dumps({"data": {"test": "value"}})
    mock_graphql.return_value = mock_graphql_instance
    return mock_graphql


@pytest.fixture
def sample_shopify_product_data() -> dict[str, Any]:
    """Provide sample Shopify product data for testing."""
    return {
        "data": {
            "products": {
                "nodes": [
                    {
                        "id": "gid://shopify/Product/12345",
                        "title": "Test Product",
                        "description": "A test product description",
                        "handle": "test-product",
                        "onlineStoreUrl": "https://test-shop.myshopify.com/products/test-product",
                        "images": {
                            "edges": [
                                {
                                    "node": {
                                        "src": "https://cdn.shopify.com/test-image.jpg",
                                        "altText": "Test Product Image",
                                    }
                                }
                            ]
                        },
                        "variants": {
                            "nodes": [
                                {
                                    "displayName": "Small",
                                    "id": "gid://shopify/ProductVariant/67890",
                                    "price": "29.99",
                                    "inventoryQuantity": 10,
                                }
                            ]
                        },
                    }
                ],
                "pageInfo": {
                    "endCursor": "cursor123",
                    "hasNextPage": False,
                    "hasPreviousPage": False,
                    "startCursor": "cursor456",
                },
            }
        }
    }


# HubSpot-specific fixtures
@pytest.fixture
def mock_hubspot_client() -> MagicMock:
    """Create a mock HubSpot client with predefined responses."""
    mock_client = MagicMock()

    # Mock successful contact search response
    mock_search_response = MagicMock()
    mock_search_response.to_dict.return_value = {
        "total": 1,
        "results": [
            {
                "id": "12345",
                "properties": {
                    "firstname": "John",
                    "lastname": "Doe",
                    "email": "john.doe@example.com",
                },
            }
        ],
    }
    mock_client.crm.contacts.search_api.do_search.return_value = mock_search_response

    # Mock communication record creation
    mock_comm_response = MagicMock()
    mock_comm_response.to_dict.return_value = {"id": "comm_123"}
    mock_client.crm.objects.communications.basic_api.create.return_value = (
        mock_comm_response
    )

    # Mock contact-communication association
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


@pytest.fixture
def mock_hubspot_ticket_client() -> MagicMock:
    """Create a mock HubSpot client with ticket creation responses."""
    mock_client = MagicMock()

    # Mock successful ticket creation
    mock_ticket_response = MagicMock()
    mock_ticket_response.to_dict.return_value = {"id": "ticket_123"}
    mock_client.crm.tickets.basic_api.create.return_value = mock_ticket_response

    # Mock ticket-contact association
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


@pytest.fixture
def mock_hubspot_meeting_client() -> MagicMock:
    """Create a mock HubSpot client with meeting creation responses."""
    mock_client = MagicMock()

    # Mock successful meeting creation
    mock_meeting_response = MagicMock()
    mock_meeting_response.to_dict.return_value = {"id": "meeting_123"}
    mock_client.crm.objects.meetings.basic_api.create.return_value = (
        mock_meeting_response
    )

    # Mock meeting-contact association
    mock_client.crm.associations.v4.basic_api.create.return_value = None

    return mock_client


# HITL-specific fixtures
@pytest.fixture
def mock_message_state() -> Mock:
    """Create a mock MessageState for HITL testing."""
    from arklex.utils.graph_state import (
        BotConfig,
        ConvoMessage,
        LLMConfig,
        MessageState,
        Metadata,
        OrchestratorMessage,
        StatusEnum,
        Timing,
    )

    return MessageState(
        sys_instruct="Mock system instructions",
        bot_config=BotConfig(
            bot_id="test",
            version="1.0",
            language="EN",
            bot_type="test",
            llm_config=LLMConfig(
                model_type_or_path="gpt-3.5-turbo", llm_provider="openai"
            ),
        ),
        user_message=ConvoMessage(
            history="Mock conversation history", message="Mock user message"
        ),
        orchestrator_message=OrchestratorMessage(
            message="Mock orchestrator message", attribute={}
        ),
        function_calling_trajectory=[],
        trajectory=[],
        message_flow="Mock message flow",
        response="Mock response",
        status=StatusEnum.COMPLETE,
        slots={},
        metadata=Metadata(
            chat_id="test-chat-id",
            turn_id=1,
            hitl=None,
            timing=Timing(),
            attempts=None,
        ),
        is_stream=False,
        message_queue=None,
        stream_type=None,
        relevant_records=None,
    )


@pytest.fixture
def mock_llm_response() -> Mock:
    """Create a mock LLM response for testing."""
    mock_response = Mock()
    mock_response.content = "Mock LLM response content"
    return mock_response


@pytest.fixture
def mock_embeddings_response() -> list[list[float]]:
    """Create mock embeddings response for testing."""
    return [[0.1] * 1536] * 5  # Mock 5 documents with 1536-dimensional vectors


# Common test utilities
@pytest.fixture
def sample_conversation_history() -> list[dict[str, str]]:
    """Provide sample conversation history for testing."""
    return [
        {"role": "user", "content": "Hello, I need help with my order"},
        {
            "role": "assistant",
            "content": "I'd be happy to help you with your order. What's your order number?",
        },
        {"role": "user", "content": "My order number is 12345"},
    ]


@pytest.fixture
def sample_user_parameters() -> dict[str, Any]:
    """Provide sample user parameters for testing."""
    return {
        "order_id": "12345",
        "email": "test@example.com",
        "product_query": "shoes",
        "chat_id": "test-chat-123",
    }


def pytest_configure(config: pytest.Config) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "hitl: marks tests as HITL (Human-in-the-Loop) tests"
    )
    config.addinivalue_line("markers", "slow: marks tests as slow running tests")
    config.addinivalue_line(
        "markers", "shopify: marks tests as Shopify integration tests"
    )
    config.addinivalue_line(
        "markers", "hubspot: marks tests as HubSpot integration tests"
    )


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

        # Mark Shopify tests
        if "shopify" in str(item.fspath) or "shopify" in item.name:
            item.add_marker(pytest.mark.shopify)

        # Mark HubSpot tests
        if "hubspot" in str(item.fspath) or "hubspot" in item.name:
            item.add_marker(pytest.mark.hubspot)

        # Mark slow tests (can be customized based on test names)
        if any(
            slow_indicator in item.name.lower()
            for slow_indicator in ["slow", "integration", "e2e"]
        ):
            item.add_marker(pytest.mark.slow)
