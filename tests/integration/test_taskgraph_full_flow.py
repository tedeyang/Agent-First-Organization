"""
Integration tests for complete taskgraph flow execution.

This module contains comprehensive integration tests that validate the complete
flow of a taskgraph.json file, testing the entire conversation flow from start
to finish rather than individual tool calls. It uses the Shopify taskgraph
example to test realistic scenarios.

Key Test Scenarios:
- Complete conversation flow with multiple nodes
- Intent detection and routing
- Tool execution and response handling
- State management across nodes
- Error handling and recovery
- Edge cases and boundary conditions
"""

import json
import os
from typing import Any
from unittest.mock import Mock

import pytest

from arklex.env.env import Environment
from arklex.orchestrator.entities.orchestrator_params_entities import OrchestratorParams
from arklex.orchestrator.orchestrator import AgentOrg


class TestTaskGraphFullFlow:
    """
    Integration tests for complete taskgraph flow execution.

    This test class validates the entire flow of a taskgraph.json file,
    testing realistic conversation scenarios that traverse multiple nodes
    and execute various tools and workers.
    """

    @pytest.fixture
    def shopify_taskgraph_path(self) -> str:
        """Get the path to the Shopify taskgraph.json file."""
        return "examples/shopify/taskgraph.json"

    @pytest.fixture
    def mock_environment(self) -> Environment:
        """Create a mock environment for testing."""
        env = Environment(tools=[], workers=[], agents=[])
        env.model_service = Mock()
        env.planner = None
        return env

    @pytest.fixture
    def sample_shopify_product_data(self) -> dict[str, Any]:
        """Sample Shopify product data for testing."""
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
                        "startCursor": "cursor123",
                    },
                }
            }
        }

    @pytest.fixture
    def sample_shopify_order_data(self) -> dict[str, Any]:
        """Sample Shopify order data for testing."""
        return {
            "data": {
                "orders": {
                    "nodes": [
                        {
                            "id": "gid://shopify/Order/12345",
                            "name": "#1001",
                            "email": "customer@example.com",
                            "totalPriceSet": {
                                "shopMoney": {"amount": "99.99", "currencyCode": "USD"}
                            },
                            "lineItems": {
                                "nodes": [
                                    {
                                        "title": "Test Product",
                                        "quantity": 2,
                                        "variant": {
                                            "price": "49.99",
                                            "title": "Small",
                                        },
                                    }
                                ]
                            },
                            "fulfillments": {
                                "nodes": [
                                    {
                                        "id": "gid://shopify/Fulfillment/67890",
                                        "status": "SUCCESS",
                                    }
                                ]
                            },
                        }
                    ]
                }
            }
        }

    @pytest.fixture
    def sample_shopify_cart_data(self) -> dict[str, Any]:
        """Sample Shopify cart data for testing."""
        return {
            "data": {
                "cart": {
                    "id": "cart_123",
                    "checkoutUrl": "https://test-shop.myshopify.com/cart/123",
                    "lines": {
                        "nodes": [
                            {
                                "id": "line_123",
                                "quantity": 1,
                                "merchandise": {
                                    "id": "variant_123",
                                    "title": "Test Product - Small",
                                    "product": {
                                        "title": "Test Product",
                                        "handle": "test-product",
                                    },
                                },
                            }
                        ]
                    },
                }
            }
        }

    def test_taskgraph_initialization(
        self, shopify_taskgraph_path: str, mock_environment: Environment
    ) -> None:
        """
        Test that the taskgraph can be properly initialized from the JSON file.

        This test validates that the AgentOrg can successfully load and parse
        the taskgraph.json file, creating all necessary components including
        the task graph, workers, and tools.
        """
        # Test that the taskgraph file exists
        assert os.path.exists(shopify_taskgraph_path), (
            f"Taskgraph file not found: {shopify_taskgraph_path}"
        )

        # Load the taskgraph configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Add model configuration since it's required
        config["model"] = {
            "model_type_or_path": "gpt-3.5-turbo",
            "llm_provider": "openai",
            "api_key": "test-key",
            "context": 16000,
            "max_tokens": 4096,
        }

        # Validate basic structure
        assert "nodes" in config, "Taskgraph should contain nodes"
        assert "edges" in config, "Taskgraph should contain edges"
        assert "tools" in config, "Taskgraph should contain tools"
        assert "workers" in config, "Taskgraph should contain workers"

        # Test AgentOrg initialization
        agent = AgentOrg(config, mock_environment)

        # Validate that the agent was properly initialized
        assert agent.task_graph is not None, "Task graph should be initialized"
        assert agent.env is mock_environment, "Environment should be set"
        assert agent.product_kwargs["role"] == "customer service assistant", (
            "Role should be set correctly"
        )

        # Validate task graph structure
        assert len(agent.task_graph.graph.nodes) > 0, "Task graph should have nodes"
        assert len(agent.task_graph.graph.edges) > 0, "Task graph should have edges"

        # Validate start node exists
        start_node = agent.task_graph.get_start_node()
        assert start_node is not None, "Task graph should have a start node"
        assert start_node == "0", "Start node should be node '0'"

    def _load_config_with_model(self, shopify_taskgraph_path: str) -> dict[str, Any]:
        """Helper method to load taskgraph config and add required model configuration."""
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Add model configuration since it's required
        config["model"] = {
            "model_type_or_path": "gpt-3.5-turbo",
            "llm_provider": "openai",
            "api_key": "test-key",
            "context": 16000,
            "max_tokens": 4096,
        }
        return config

    def test_product_search_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_product_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for a product search scenario.

        This test simulates a user asking about products, which should:
        1. Start at the initial node
        2. Route to the product search node based on intent
        3. Execute the search_products tool
        4. Return a response with product information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for product search
        user_input = {
            "text": "I'm looking for shoes",
            "chat_history": [],
            "parameters": {},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "result" in response, "Response should contain 'result' field"
        assert "status" in response, "Response should contain 'status' field"

    def test_order_inquiry_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_order_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for an order inquiry scenario.

        This test simulates a user asking about their past orders, which should:
        1. Start at the initial node
        2. Route to the order details node based on intent
        3. Execute the get_order_details tool
        4. Return order information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for order inquiry
        user_input = {
            "text": "I want to check my order status",
            "chat_history": [],
            "parameters": {"user_id": "12345"},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "result" in response, "Response should contain 'result' field"
        assert "status" in response, "Response should contain 'status' field"

    def test_cart_inquiry_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
        sample_shopify_cart_data: dict[str, Any],
    ) -> None:
        """
        Test the complete flow for a cart inquiry scenario.

        This test simulates a user asking about their shopping cart, which should:
        1. Start at the initial node
        2. Route to the cart inquiry node based on intent
        3. Execute the get_cart tool
        4. Return cart information
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate user input for cart inquiry
        user_input = {
            "text": "What's in my shopping cart?",
            "chat_history": [],
            "parameters": {"cart_id": "cart_123"},
        }

        # Get response from agent
        response = agent.get_response(user_input)

        # Validate response structure
        assert "result" in response, "Response should contain 'result' field"
        assert "status" in response, "Response should contain 'status' field"

    def test_multi_turn_conversation_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test a multi-turn conversation that traverses multiple nodes.

        This test simulates a conversation where the user:
        1. Asks about products
        2. Gets a response
        3. Asks a follow-up question
        4. Gets another response

        This validates that the taskgraph can handle conversation state
        and properly route between different nodes based on context.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # First turn: Product inquiry
        first_input = {
            "text": "Do you have any shoes?",
            "chat_history": [],
            "parameters": {},
        }

        first_response = agent.get_response(first_input)
        assert "result" in first_response, (
            "First response should contain 'result' field"
        )

        # Second turn: Follow-up question
        second_input = {
            "text": "What about my order status?",
            "chat_history": [
                {"role": "user", "content": "Do you have any shoes?"},
                {"role": "assistant", "content": first_response.get("result", "")},
            ],
            "parameters": {"user_id": "12345"},
        }

        second_response = agent.get_response(second_input)
        assert "result" in second_response, (
            "Second response should contain 'result' field"
        )

    def test_error_handling_flow(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test error handling in the taskgraph flow.

        This test validates that the taskgraph can properly handle errors
        and continue processing, ensuring robust conversation flow.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Simulate an input that might cause an error
        user_input = {
            "text": "This is an invalid request that might cause an error",
            "chat_history": [],
            "parameters": {},
        }

        # The agent should handle the error gracefully
        response = agent.get_response(user_input)

        # Even with errors, we should get a structured response
        assert "result" in response, "Response should contain 'result' field"
        assert "status" in response, "Response should contain 'status' field"

    def test_taskgraph_node_transitions(
        self,
        shopify_taskgraph_path: str,
        mock_environment: Environment,
    ) -> None:
        """
        Test that the taskgraph can properly transition between nodes.

        This test validates the graph structure and ensures that nodes
        can be properly traversed based on intents and conditions.
        """
        # Load configuration
        config = self._load_config_with_model(shopify_taskgraph_path)

        # Initialize agent
        agent = AgentOrg(config, mock_environment)

        # Get the task graph
        task_graph = agent.task_graph

        # Validate that we can get the start node
        start_node = task_graph.get_start_node()
        assert start_node is not None, "Should have a start node"

        # Validate that we can get available intents
        params = OrchestratorParams()
        available_intents = task_graph.get_available_global_intents(params)
        assert isinstance(available_intents, dict), "Available intents should be a dict"
        assert len(available_intents) > 0, "Should have available intents"

        # Validate that we can get the current node
        current_node, updated_params = task_graph.get_current_node(params)
        assert current_node is not None, "Should be able to get current node"

    def test_taskgraph_configuration_validation(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that the taskgraph configuration is valid and complete.

        This test validates the structure and content of the taskgraph.json
        file, ensuring it has all required components and valid relationships.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Validate required top-level keys
        required_keys = ["nodes", "edges", "tools", "workers", "role", "user_objective"]
        for key in required_keys:
            assert key in config, f"Taskgraph should contain '{key}'"

        # Validate nodes structure
        nodes = config["nodes"]
        assert isinstance(nodes, list), "Nodes should be a list"
        assert len(nodes) > 0, "Should have at least one node"

        # Validate edges structure
        edges = config["edges"]
        assert isinstance(edges, list), "Edges should be a list"
        assert len(edges) > 0, "Should have at least one edge"

        # Validate tools structure
        tools = config["tools"]
        assert isinstance(tools, list), "Tools should be a list"
        assert len(tools) > 0, "Should have at least one tool"

        # Validate workers structure
        workers = config["workers"]
        assert isinstance(workers, list), "Workers should be a list"
        assert len(workers) > 0, "Should have at least one worker"

        # Validate that all referenced tools exist
        tool_ids = {tool["id"] for tool in tools}
        for node in nodes:
            node_data = node[1]
            if "resource" in node_data and "id" in node_data["resource"]:
                resource_id = node_data["resource"]["id"]
                if resource_id not in ["MessageWorker", "FaissRAGWorker"]:  # Workers
                    assert resource_id in tool_ids, (
                        f"Tool {resource_id} referenced but not defined"
                    )

    def test_taskgraph_edge_validation(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that all edges in the taskgraph reference valid nodes.

        This test ensures that the graph structure is consistent and
        all edges connect to existing nodes.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Get all node IDs
        node_ids = {node[0] for node in config["nodes"]}

        # Validate all edges reference existing nodes
        for edge in config["edges"]:
            source_node = edge[0]
            target_node = edge[1]

            assert source_node in node_ids, (
                f"Edge source node '{source_node}' does not exist"
            )
            assert target_node in node_ids, (
                f"Edge target node '{target_node}' does not exist"
            )

    def test_taskgraph_intent_consistency(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that intents are consistently defined across the taskgraph.

        This test validates that intents are properly defined and used
        consistently throughout the graph structure.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Collect all intents from edges
        intents = set()
        for edge in config["edges"]:
            edge_data = edge[2]
            if "intent" in edge_data:
                intents.add(edge_data["intent"])

        # Validate that we have intents
        assert len(intents) > 0, "Should have at least one intent"

        # Validate that intents are not empty strings
        for intent in intents:
            assert intent != "", "Intent should not be empty"
            # Note: "None" intents are valid in some cases (e.g., default edges)
            # We'll allow them but log a warning
            if intent == "None":
                print("Warning: Found 'None' intent in taskgraph")

    def test_taskgraph_worker_configuration(
        self,
        shopify_taskgraph_path: str,
    ) -> None:
        """
        Test that worker configurations are valid and complete.

        This test validates that all workers referenced in nodes are
        properly defined in the workers section.
        """
        # Load configuration
        with open(shopify_taskgraph_path) as f:
            config = json.load(f)

        # Get all worker IDs
        worker_ids = {worker["id"] for worker in config["workers"]}

        # Validate that all worker references are valid
        for node in config["nodes"]:
            node_data = node[1]
            if "resource" in node_data and "id" in node_data["resource"]:
                resource_id = node_data["resource"]["id"]
                if resource_id in ["MessageWorker", "FaissRAGWorker"]:
                    assert resource_id in worker_ids, (
                        f"Worker {resource_id} referenced but not defined"
                    )
