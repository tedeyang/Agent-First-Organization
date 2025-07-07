"""
Shared test utilities for integration tests.

This module provides common utilities, helper functions, and test data
that can be used across all integration tests.
"""

import json
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from arklex.env.env import Environment
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


class MilvusTestHelper:
    """Helper class for Milvus-specific test utilities."""

    @staticmethod
    def create_mock_message_state(response: str = "Mock response") -> "MessageState":
        """Create a mock MessageState for Milvus testing."""
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

    @staticmethod
    def get_api_bot_response(
        config: dict[str, Any],
        env: "Environment",
        user_text: str,
        history: list[dict[str, str]],
        params: dict[str, Any],
    ) -> tuple[str, dict[str, Any], str | None]:
        """Helper method to get bot response."""
        from arklex.orchestrator.orchestrator import AgentOrg

        data = {
            "text": user_text,
            "chat_history": history,
            "parameters": params,
        }
        orchestrator = AgentOrg(config=config, env=env)
        result = orchestrator.get_response(data)

        return result["answer"], result["parameters"], result["human_in_the_loop"]

    @staticmethod
    def validate_taskgraph_structure(config: dict) -> None:
        """Validate that the taskgraph has the correct structure and required fields."""
        # Check required top-level fields
        required_fields = [
            "nodes",
            "edges",
            "role",
            "user_objective",
            "domain",
            "intro",
        ]
        for field in required_fields:
            assert field in config, f"Required field '{field}' missing from taskgraph"

        # Check nodes structure
        assert isinstance(config["nodes"], list), "Nodes should be a list"
        assert len(config["nodes"]) >= 2, (
            "Should have at least 2 nodes (start + worker)"
        )

        # Check edges structure
        assert isinstance(config["edges"], list), "Edges should be a list"
        assert len(config["edges"]) >= 1, "Should have at least 1 edge"

        # Check that start node exists
        start_node_found = False
        for _node_id, node_data in config["nodes"]:
            if node_data.get("type") == "start":
                start_node_found = True
                break
        assert start_node_found, "Start node not found in taskgraph"

        # Check that MilvusRAGWorker is configured
        milvus_worker_found = False
        for _node_id, node_data in config["nodes"]:
            if (
                node_data.get("resource", {}).get("name") == "MilvusRAGWorker"
                or node_data.get("resource", {}).get("id") == "milvus_rag_worker"
            ):
                milvus_worker_found = True
                break
        assert milvus_worker_found, "MilvusRAGWorker not found in taskgraph"

    @staticmethod
    def validate_worker_configuration(config: dict) -> None:
        """Validate that workers are properly configured in the taskgraph."""
        # Check workers list
        assert "workers" in config, "Workers configuration missing"
        assert isinstance(config["workers"], list), "Workers should be a list"

        # Check required workers
        worker_names = [worker.get("name") for worker in config["workers"]]
        assert "MessageWorker" in worker_names, "MessageWorker not configured"
        assert "MilvusRAGWorker" in worker_names, "MilvusRAGWorker not configured"

        # Check worker paths
        for worker in config["workers"]:
            assert "id" in worker, f"Worker missing 'id' field: {worker}"
            assert "name" in worker, f"Worker missing 'name' field: {worker}"
            assert "path" in worker, f"Worker missing 'path' field: {worker}"

    @staticmethod
    def validate_domain_specific_configuration(config: dict) -> None:
        """Validate that the taskgraph is properly configured for robotics domain."""
        # Check domain-specific content
        assert config["domain"] == "robotics and automation", "Incorrect domain"
        assert "Richtech Robotics" in config["intro"], "Missing company information"
        assert "robots" in config["intro"].lower(), "Missing robot information"

        # Check that product tags are configured
        for _node_id, node_data in config["nodes"]:
            if node_data.get("resource", {}).get("name") == "MilvusRAGWorker":
                tags = node_data.get("attribute", {}).get("tags", {})
                assert "product" in tags, (
                    "Product tags not configured for MilvusRAGWorker"
                )
                assert tags["product"] == "robots", "Incorrect product tag value"

    @staticmethod
    def validate_taskgraph_metadata(config: dict) -> None:
        """Validate that taskgraph metadata is properly configured."""
        # Check role and objectives
        assert config["role"] == "customer service assistant", "Incorrect role"
        assert "customer service" in config["user_objective"].lower(), (
            "Missing customer service objective"
        )
        assert "contact information" in config["builder_objective"].lower(), (
            "Missing builder objective"
        )

        # Check domain-specific information
        assert "Richtech Robotics" in config["intro"], "Missing company name"
        assert "Las Vegas" in config["intro"], "Missing headquarters information"
        assert "Austin" in config["intro"], "Missing office information"
        assert "www.cloutea.com" in config["intro"], "Missing ClouTea website"

        # Check product information
        assert "ADAM" in config["intro"], "Missing ADAM robot information"
        assert "ARM" in config["intro"], "Missing ARM robot information"
        assert "ACE" in config["intro"], "Missing ACE robot information"
        assert "Matradee" in config["intro"], "Missing Matradee robot information"
        assert "DUST-E" in config["intro"], "Missing DUST-E robot information"

        # Check delivery time information
        assert "one month" in config["intro"], "Missing delivery time information"
        assert "two months" in config["intro"], "Missing cleaning robot delivery time"

    @staticmethod
    def validate_node_edge_consistency(config: dict) -> None:
        """Validate that nodes and edges are consistent and properly connected."""
        # Get all node IDs
        node_ids = [node[0] for node in config["nodes"]]

        # Check that all edges reference valid nodes
        for edge in config["edges"]:
            source_node = edge[0]
            target_node = edge[1]

            assert source_node in node_ids, (
                f"Edge source node '{source_node}' not found in nodes"
            )
            assert target_node in node_ids, (
                f"Edge target node '{target_node}' not found in nodes"
            )

        # Check that start node has outgoing edges
        start_node_id = None
        for node_id, node_data in config["nodes"]:
            if node_data.get("type") == "start":
                start_node_id = node_id
                break

        assert start_node_id is not None, "Start node not found"

        # Check that start node has outgoing edges
        start_has_outgoing = any(edge[0] == start_node_id for edge in config["edges"])
        assert start_has_outgoing, "Start node should have outgoing edges"

        # Check that worker nodes have incoming edges
        worker_node_ids = [
            node_id
            for node_id, node_data in config["nodes"]
            if node_data.get("type") != "start"
        ]

        for worker_id in worker_node_ids:
            worker_has_incoming = any(edge[1] == worker_id for edge in config["edges"])
            assert worker_has_incoming, (
                f"Worker node '{worker_id}' should have incoming edges"
            )


class MilvusMockFactory:
    """Factory for creating Milvus-specific mock objects."""

    @staticmethod
    def create_mock_milvus_retrieval_success() -> Mock:
        """Create a mock Milvus retrieval function that succeeds."""

        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify that product tags are passed correctly
            if tags is not None:
                assert "product" in tags, "Product tag should be present"
                assert tags["product"] == "robots", "Product tag should be 'robots'"

            # Add retrieved context to message state
            message_state.message_flow = (
                "Retrieved information about ADAM robot bartender"
            )
            return message_state

        mock_retrieve = Mock()
        mock_retrieve.side_effect = mock_milvus_retrieve_side_effect
        return mock_retrieve

    @staticmethod
    def create_mock_milvus_retrieval_error() -> Mock:
        """Create a mock Milvus retrieval function that raises an exception."""

        def mock_milvus_retrieve_error_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            raise Exception("Milvus connection error")

        mock_retrieve = Mock()
        mock_retrieve.side_effect = mock_milvus_retrieve_error_side_effect
        return mock_retrieve

    @staticmethod
    def create_mock_context_generator(response: str = "Mock response") -> Mock:
        """Create a mock context generator."""

        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = response
            return message_state

        mock_generator = Mock()
        mock_generator.side_effect = mock_context_generate_side_effect
        return mock_generator

    @staticmethod
    def create_mock_post_process() -> Mock:
        """Create a mock post-process function."""

        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process = Mock()
        mock_post_process.side_effect = mock_post_process_side_effect
        return mock_post_process
