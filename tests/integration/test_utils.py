"""
Shared test utilities for integration tests.

This module provides common utilities, helper functions, and test data
that can be used across all integration tests.
"""

import json
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from arklex.utils.graph_state import MessageState


class MockResponse:
    """Mock response object for testing HTTP requests."""

    def __init__(
        self, status_code: int = 200, json_data: dict[str, Any] = None, text: str = ""
    ) -> None:
        self.status_code = status_code
        self._json_data = json_data or {}
        self.text = text

    def json(self) -> dict[str, Any]:
        return self._json_data


class TestDataProvider:
    """Provides common test data for integration tests."""

    @staticmethod
    def get_sample_shopify_product() -> dict[str, Any]:
        """Get sample Shopify product data."""
        return {
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

    @staticmethod
    def get_sample_hubspot_contact() -> dict[str, Any]:
        """Get sample HubSpot contact data."""
        return {
            "id": "12345",
            "properties": {
                "firstname": "John",
                "lastname": "Doe",
                "email": "john.doe@example.com",
                "phone": "+1234567890",
            },
        }

    @staticmethod
    def get_sample_conversation_history() -> list[dict[str, str]]:
        """Get sample conversation history."""
        return [
            {"role": "user", "content": "Hello, I need help with my order"},
            {
                "role": "assistant",
                "content": "I'd be happy to help you with your order. What's your order number?",
            },
            {"role": "user", "content": "My order number is 12345"},
        ]

    @staticmethod
    def get_sample_user_parameters() -> dict[str, Any]:
        """Get sample user parameters."""
        return {
            "order_id": "12345",
            "email": "test@example.com",
            "product_query": "shoes",
            "chat_id": "test-chat-123",
        }


class MockFactory:
    """Factory for creating common mock objects."""

    @staticmethod
    def create_mock_llm_response(content: str = "Mock LLM response") -> Mock:
        """Create a mock LLM response."""
        mock_response = Mock()
        mock_response.content = content
        return mock_response

    @staticmethod
    def create_mock_embeddings_response(
        dimensions: int = 1536, num_docs: int = 5
    ) -> list[list[float]]:
        """Create mock embeddings response."""
        return [[0.1] * dimensions] * num_docs

    @staticmethod
    def create_mock_http_response(
        status_code: int = 200, json_data: dict[str, Any] = None
    ) -> MockResponse:
        """Create a mock HTTP response."""
        return MockResponse(status_code=status_code, json_data=json_data)

    @staticmethod
    def create_mock_shopify_session() -> Mock:
        """Create a mock Shopify session."""
        mock_session = Mock()
        mock_session.__enter__ = Mock(return_value=mock_session)
        mock_session.__exit__ = Mock(return_value=None)
        return mock_session

    @staticmethod
    def create_mock_hubspot_client() -> MagicMock:
        """Create a mock HubSpot client."""
        mock_client = MagicMock()

        # Mock successful contact search response
        mock_search_response = MagicMock()
        mock_search_response.to_dict.return_value = {
            "total": 1,
            "results": [TestDataProvider.get_sample_hubspot_contact()],
        }
        mock_client.crm.contacts.search_api.do_search.return_value = (
            mock_search_response
        )

        # Mock communication record creation
        mock_comm_response = MagicMock()
        mock_comm_response.to_dict.return_value = {"id": "comm_123"}
        mock_client.crm.objects.communications.basic_api.create.return_value = (
            mock_comm_response
        )

        # Mock contact-communication association
        mock_client.crm.associations.v4.basic_api.create.return_value = None

        return mock_client


class AssertionHelper:
    """Helper class for common test assertions."""

    @staticmethod
    def assert_json_response_structure(response: str, expected_keys: list[str]) -> None:
        """Assert that a JSON response has the expected structure."""
        try:
            response_data = json.loads(response)
            for key in expected_keys:
                assert key in response_data, (
                    f"Expected key '{key}' not found in response"
                )
        except json.JSONDecodeError:
            pytest.fail("Response is not valid JSON")

    @staticmethod
    def assert_error_message_contains(error: Exception, expected_text: str) -> None:
        """Assert that an error message contains expected text."""
        assert expected_text in str(error), (
            f"Expected '{expected_text}' in error message, got: {str(error)}"
        )

    @staticmethod
    def assert_api_calls_made(mock_client: Mock, expected_calls: list[str]) -> None:
        """Assert that specific API calls were made."""
        for call_name in expected_calls:
            # This is a simplified check - in practice you'd want more specific assertions
            assert hasattr(mock_client, call_name), (
                f"Expected API call '{call_name}' not found"
            )


class TestEnvironmentHelper:
    """Helper for managing test environment."""

    @staticmethod
    def setup_test_environment() -> None:
        """Set up common test environment variables."""
        import os

        # These should already be set in conftest.py, but this provides a backup
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
            os.environ.setdefault(key, value)

    @staticmethod
    def cleanup_test_environment() -> None:
        """Clean up test environment (if needed)."""
        # Add any cleanup logic here
        pass


# Convenience functions for common test patterns
def create_mock_message_state(response: str = "Mock response") -> "MessageState":
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
        response=response,
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


def create_mock_shopify_graphql_response(data: dict[str, Any]) -> str:
    """Create a mock Shopify GraphQL response."""
    return json.dumps({"data": data})


def create_mock_hubspot_api_response(data: dict[str, Any]) -> MagicMock:
    """Create a mock HubSpot API response."""
    mock_response = MagicMock()
    mock_response.to_dict.return_value = data
    return mock_response
