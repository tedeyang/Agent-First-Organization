"""
Integration tests for Milvus filter taskgraph.

This module contains comprehensive integration tests for the Milvus filter taskgraph,
including proper mocking of external services, RAG functionality with product filtering,
node transitions, and edge case testing.
"""

from typing import Any
from unittest.mock import Mock, patch

from arklex.utils.graph_state import MessageState
from tests.integration.test_utils import MilvusTestHelper


class TestMilvusFilterIntegration:
    """Integration tests for Milvus filter taskgraph."""

    def test_taskgraph_structure_validation(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """Test that the taskgraph has the correct structure and required fields."""
        MilvusTestHelper.validate_taskgraph_structure(load_milvus_config)

    def test_worker_configuration(self, load_milvus_config: dict[str, Any]) -> None:
        """Test that workers are properly configured in the taskgraph."""
        MilvusTestHelper.validate_worker_configuration(load_milvus_config)

    def test_domain_specific_configuration(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """Test that the taskgraph is properly configured for robotics domain."""
        MilvusTestHelper.validate_domain_specific_configuration(load_milvus_config)

    @patch("arklex.env.env.Environment.step")
    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_start_message_delivery(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        mock_env_step: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test that the start message is delivered correctly."""
        config, env, start_message = config_and_env

        # Mock intent detection to stay at start node
        mock_intent_detector.return_value = "0) Stay at start"

        # Mock the provider map to return a mock ChatOpenAI
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = start_message
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock ToolGenerator to return proper MessageState
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = start_message
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock post-processing
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state(
                    start_message
                )
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Mock environment step to return the start message
        def mock_env_step_side_effect(
            resource_id: str,
            message_state: MessageState,
            params: dict[str, Any],
            node_info: dict[str, Any],
        ) -> tuple[MessageState, dict[str, Any]]:
            # Set the response to the start message
            message_state.response = start_message
            return message_state, params

        mock_env_step.side_effect = mock_env_step_side_effect

        # Test start message
        history: list[dict[str, str]] = []
        params: dict[str, Any] = {}
        user_text = "<start>"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify start message
        assert start_message in output, "Start message not delivered correctly"
        assert hitl is None or hitl == "", "HITL should not be active for start message"

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("arklex.utils.model_provider_config.PROVIDER_MAP")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_milvus_rag_worker_product_query(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test MilvusRAGWorker responding to product questions with filtering."""
        config, env, start_message = config_and_env

        # Mock intent detection to transition to MilvusRAGWorker
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval with product filtering
        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Verify that product tags are passed correctly
            assert tags is not None, "Tags should be passed to Milvus retrieval"
            assert "product" in tags, "Product tag should be present"
            assert tags["product"] == "robots", "Product tag should be 'robots'"

            # Add retrieved context to message state
            message_state.message_flow = (
                "Retrieved information about ADAM robot bartender"
            )
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        # Mock ToolGenerator to generate response
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "Based on the retrieved information, the ADAM robot is a bartender "
                "that makes tea, coffee, and cocktails. It's available for both "
                "purchase and rental for multiple purposes."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock the provider map to return a mock ChatOpenAI
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "Product information response"
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock post-processing
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test product query
        history: list[dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]
        params: dict[str, Any] = {}
        user_text = "Tell me about your ADAM robot"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify response contains product information
        assert "ADAM" in output, "Response should mention ADAM robot"
        assert "bartender" in output.lower(), (
            "Response should mention bartender functionality"
        )
        assert hitl is None or hitl == "", (
            "HITL should not be active for product queries"
        )

        # Verify node transition
        assert params["taskgraph"]["curr_node"] == "1", (
            "Should transition to MilvusRAGWorker node"
        )

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_conversation_flow_multiple_turns(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test conversation flow through multiple turns with different queries."""
        config, env, start_message = config_and_env

        # First turn - product query
        mock_intent_detector.return_value = "1) User seeks information about robots"

        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            message_state.message_flow = "Retrieved information about delivery robots"
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "Our delivery robots include Matradee, Matradee X, and Matradee L. "
                "Delivery time is typically one month for delivery robots."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock the provider map to return a mock ChatOpenAI
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "Delivery robot information"
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # First turn
        history: list[dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]
        params: dict[str, Any] = {}
        user_text = "What delivery robots do you have?"

        output1, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify first response
        assert "Matradee" in output1, "First response should mention Matradee"
        assert "delivery" in output1.lower(), "First response should mention delivery"

        # Second turn - different product query
        mock_intent_detector.return_value = "1) User seeks information about robots"

        def mock_milvus_retrieve_side_effect_second(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            message_state.message_flow = "Retrieved information about cleaning robots"
            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect_second

        def mock_context_generate_side_effect_second(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "Our cleaning robots include DUST-E SX and DUST-E MX. "
                "Delivery time is two months for commercial cleaning robots."
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect_second

        # Second turn
        history = [
            {"role": "assistant", "content": start_message},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output1},
        ]
        user_text2 = "Tell me about your cleaning robots"

        output2, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text2, history, params
        )

        # Verify second response
        assert "DUST-E" in output2, "Second response should mention DUST-E"
        assert "cleaning" in output2.lower(), "Second response should mention cleaning"

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_edge_case_empty_input(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test handling of empty or invalid input."""
        config, env, start_message = config_and_env

        # Mock intent detection for empty input
        mock_intent_detector.return_value = "0) Stay at start"

        # Mock the provider map to return a mock ChatOpenAI
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = (
            "I didn't understand that. Could you please rephrase your question?"
        )
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock ToolGenerator to return proper MessageState
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = (
                "I didn't understand that. Could you please rephrase your question?"
            )
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock post-processing
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test empty input
        history: list[dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]
        params: dict[str, Any] = {}
        user_text = ""

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify graceful handling
        assert len(output) > 0, "Should provide a response even for empty input"
        assert hitl is None or hitl == "", "HITL should not be active for empty input"

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_milvus_retrieval_error_handling(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_provider_map: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test handling of Milvus retrieval errors."""
        config, env, start_message = config_and_env

        # Mock intent detection
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Mock Milvus retrieval to raise an exception
        mock_milvus_retrieve.side_effect = Exception("Milvus connection error")

        # Mock the provider map to return a mock ChatOpenAI
        mock_chat = Mock()
        mock_response = Mock()
        mock_response.content = "I'm sorry, I'm having trouble accessing the product information right now. Please try again later."
        mock_chat.return_value = mock_response
        mock_provider_map.__getitem__.return_value = mock_chat

        # Mock embeddings
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        # Mock ToolGenerator to return proper MessageState
        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = "I'm sorry, I'm having trouble accessing the product information right now. Please try again later."
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock post-processing
        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test error handling
        history: list[dict[str, str]] = [
            {"role": "assistant", "content": start_message}
        ]
        params: dict[str, Any] = {}
        user_text = "Tell me about your robots"

        output, params, hitl = MilvusTestHelper.get_api_bot_response(
            config, env, user_text, history, params
        )

        # Verify error handling
        assert "sorry" in output.lower() or "trouble" in output.lower(), (
            "Should provide error message"
        )
        assert hitl is None or hitl == "", "HITL should not be active for errors"

    @patch("arklex.orchestrator.orchestrator.post_process_response")
    @patch("langchain_openai.embeddings.base.OpenAIEmbeddings.embed_documents")
    @patch("langchain_openai.chat_models.base.ChatOpenAI")
    @patch(
        "arklex.env.tools.RAG.retrievers.milvus_retriever.RetrieveEngine.milvus_retrieve"
    )
    @patch("arklex.env.tools.utils.ToolGenerator.context_generate")
    @patch("arklex.orchestrator.NLU.core.intent.IntentDetector.execute")
    @patch("arklex.env.planner.react_planner.FAISS")
    @patch("arklex.env.tools.RAG.retrievers.faiss_retriever.FAISS")
    def test_product_filtering_accuracy(
        self,
        mock_faiss_retriever: Mock,
        mock_faiss_planner: Mock,
        mock_intent_detector: Mock,
        mock_generator: Mock,
        mock_milvus_retrieve: Mock,
        mock_chat: Mock,
        mock_embeddings: Mock,
        mock_post_process: Mock,
        config_and_env: tuple[dict[str, Any], Any, str],
    ) -> None:
        """Test that product filtering works correctly for different queries."""
        config, env, start_message = config_and_env

        # Mock intent detection
        mock_intent_detector.return_value = "1) User seeks information about robots"

        # Track calls to verify filtering
        retrieval_calls = []

        def mock_milvus_retrieve_side_effect(
            message_state: MessageState, tags: dict[str, str] | None = None
        ) -> MessageState:
            # Record the tags used for filtering
            retrieval_calls.append(tags)

            # Simulate different responses based on query context
            user_message = message_state.user_message.message.lower()
            if "adam" in user_message:
                message_state.message_flow = (
                    "Retrieved ADAM robot bartender information"
                )
            elif "delivery" in user_message:
                message_state.message_flow = "Retrieved delivery robot information"
            elif "cleaning" in user_message:
                message_state.message_flow = "Retrieved cleaning robot information"
            else:
                message_state.message_flow = "Retrieved general robot information"

            return message_state

        mock_milvus_retrieve.side_effect = mock_milvus_retrieve_side_effect

        def mock_context_generate_side_effect(
            message_state: MessageState,
        ) -> MessageState:
            message_state.response = f"Response based on: {message_state.message_flow}"
            return message_state

        mock_generator.side_effect = mock_context_generate_side_effect

        # Mock other dependencies
        mock_response = Mock()
        mock_response.content = "Product information"
        mock_chat.return_value = mock_response
        mock_embeddings.return_value = [[0.1] * 1536] * 3

        # Mock FAISS operations
        mock_vectorstore = Mock()
        mock_vectorstore.similarity_search_with_score.return_value = [
            (Mock(metadata={"resource_name": "test_resource"}), 0.8)
        ]
        mock_faiss_planner.return_value = mock_vectorstore
        mock_faiss_retriever.return_value = mock_vectorstore

        def mock_post_process_side_effect(
            message_state: MessageState | None,
            params: dict[str, Any],
            hitl_available: bool,
            hitl_enabled: bool,
        ) -> MessageState:
            if message_state is None:
                message_state = MilvusTestHelper.create_mock_message_state()
            return message_state

        mock_post_process.side_effect = mock_post_process_side_effect

        # Test different product queries
        history = [{"role": "assistant", "content": start_message}]
        params = {}

        queries = [
            "Tell me about the ADAM robot",
            "What delivery robots do you have?",
            "Tell me about your cleaning robots",
        ]

        for query in queries:
            output, params, hitl = MilvusTestHelper.get_api_bot_response(
                config, env, query, history, params
            )

            # Verify that product filtering was applied
            assert len(retrieval_calls) > 0, "Retrieval should be called"
            last_call = retrieval_calls[-1]
            assert last_call is not None, "Tags should be passed"
            assert "product" in last_call, "Product tag should be present"
            assert last_call["product"] == "robots", "Product tag should be 'robots'"

    def test_environment_initialization(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """Test that the environment initializes correctly with Milvus workers."""
        from arklex.env.env import Environment
        from arklex.orchestrator.NLU.services.model_service import ModelService

        config = load_milvus_config

        # Initialize model service
        model_service = ModelService(config["model"])

        # Initialize environment
        env = Environment(
            tools=config.get("tools", []),
            workers=config.get("workers", []),
            agents=config.get("agents", []),
            slot_fill_api=config["slotfillapi"],
            planner_enabled=True,
            model_service=model_service,
        )

        # Verify environment components
        assert env is not None, "Environment should be created"
        assert hasattr(env, "workers"), "Environment should have workers"
        assert hasattr(env, "tools"), "Environment should have tools"

        # Verify that required workers are available
        worker_names = [worker.get("name") for worker in config["workers"]]
        assert "MessageWorker" in worker_names, "MessageWorker should be available"
        assert "MilvusRAGWorker" in worker_names, "MilvusRAGWorker should be available"

    def test_taskgraph_metadata_validation(
        self, load_milvus_config: dict[str, Any]
    ) -> None:
        """Test that taskgraph metadata is properly configured."""
        MilvusTestHelper.validate_taskgraph_metadata(load_milvus_config)

    def test_node_edge_consistency(self, load_milvus_config: dict[str, Any]) -> None:
        """Test that nodes and edges are consistent and properly connected."""
        MilvusTestHelper.validate_node_edge_consistency(load_milvus_config)
